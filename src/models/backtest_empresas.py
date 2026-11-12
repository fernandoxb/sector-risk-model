"""Modelo de risco por empresa: o backtest e a previsão oficial dos setores, aplicados às empresas.

Reutiliza `backtest.run` sem lógica nova: histórico, GJR-GARCH(1,1)-t e FHS, previstos só com dados até à
véspera, com os mesmos testes (Kupiec, Christoffersen, Z2, semáforo). Entradas: data/clean/prices_empresas.parquet.
Saídas em data/features:
  company_backtest.csv   uma linha por empresa × método × nível (igual a var_backtest.csv)
  company_risk.csv       previsão oficial (FHS) para a próxima sessão: var99, var975, es975, semáforo a 99%,
                         validado (passa Kupiec, Christoffersen e Z2 a 97,5%), es_subestimado (Z2 rejeita, Z2 < 0),
                         mult_es (correção k = 1 − Z2, limitada) e es975_aj = es975 × mult_es
O backtest completo corre na última sessão de cada semana; nos outros dias só se refaz a previsão (daily_forecast).
Empresas com menos de COMPANY_MIN_DAYS de histórico ficam sem modelo (continuam com o risco descritivo).

Funções puras: com_historico, resumo_metodos, validado, es_multiplier, flags, with_flags, daily_forecast,
needs_full, portfolio_forecast, portfolio_backtest.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (BT_SIGNIF, BT_WINDOW, CLEAN, COMPANY_FULL_WEEKLY, COMPANY_MIN_DAYS,  # noqa: E402
                             CVAR_LEVEL, ES_ADJ_MAX, FEATURES, MODEL_RISK_LEVEL, OFFICIAL_VAR_METHOD)
from src.collect.sessao import ultima_sessao_da_semana  # noqa: E402
from src.models import backtest as bt  # noqa: E402

TESTES = {"Kupiec": "p_kupiec", "Christoffersen": "p_christoffersen", "Condicional": "p_cc", "Z2 (ES)": "p_z2"}


def com_historico(prices: pd.DataFrame, min_days: int = COMPANY_MIN_DAYS) -> list[str]:
    """Empresas com pelo menos `min_days` preços (as outras não têm janela para o modelo)."""
    return [t for t in prices.columns if prices[t].notna().sum() >= min_days]


def resumo_metodos(res: pd.DataFrame, signif: float = BT_SIGNIF) -> pd.DataFrame:
    """Nº de séries em que cada teste NÃO rejeita o modelo, por método e nível (+ total de séries)."""
    ok = res.assign(**{nome: res[col] >= signif for nome, col in TESTES.items()})
    out = ok.groupby(["metodo", "nivel"])[list(TESTES)].sum()
    out["series"] = ok.groupby(["metodo", "nivel"]).size()
    return out.reset_index()


def validado(res: pd.DataFrame, level: float, method: str = OFFICIAL_VAR_METHOD, signif: float = BT_SIGNIF) -> pd.Series:
    """Por série: o método passa a frequência (Kupiec), a independência (Christoffersen) e o ES (Z2) ao nível dado?"""
    x = res[(res["metodo"] == method) & np.isclose(res["nivel"], level)].set_index(res.columns[0])
    return ((x["p_kupiec"] >= signif) & (x["p_christoffersen"] >= signif) & (x["p_z2"] >= signif)).rename("validado")


def es_multiplier(z2: float, p_z2: float, signif: float = BT_SIGNIF, cap: float = ES_ADJ_MAX) -> float:
    """Fator de correção do ES: 1 se o Z2 não rejeita; senão k = 1 − Z2 (o que teria dado Z2 = 0 no backtest),
    limitado a `cap`. Só agrava: Z2 >= 0 nunca reduz o ES."""
    if not (p_z2 < signif and z2 < 0):
        return 1.0
    return float(min(1.0 - z2, cap))


def flags(res: pd.DataFrame, level: float = CVAR_LEVEL, method: str = OFFICIAL_VAR_METHOD) -> pd.DataFrame:
    """Por série (1.ª coluna de `res`): semáforo a 99%, validado, es_subestimado e mult_es, do backtest oficial."""
    chave = res.columns[0]
    z = res[(res["metodo"] == method) & np.isclose(res["nivel"], level)].set_index(chave)
    sem = res[(res["metodo"] == method) & np.isclose(res["nivel"], MODEL_RISK_LEVEL)].set_index(chave)["semaforo"]
    return pd.DataFrame({
        "semaforo": sem, "validado": validado(res, level, method),
        "es_subestimado": (z["p_z2"] < BT_SIGNIF) & (z["z2"] < 0),
        "mult_es": [es_multiplier(a, b) for a, b in zip(z["z2"], z["p_z2"])],
    }, index=z.index)


def with_flags(official: pd.DataFrame, fl: pd.DataFrame) -> pd.DataFrame:
    """Junta as marcas do backtest à previsão e calcula o ES corrigido (es975_aj = es975 × mult_es)."""
    out = official.join(fl, on=official.columns[0])
    out["mult_es"] = out["mult_es"].fillna(1.0)
    out["es975_aj"] = out["es975"] * out["mult_es"]
    return out


def daily_forecast(prices: pd.DataFrame, tickers: list[str], window: int = BT_WINDOW) -> pd.DataFrame:
    """Só a previsão oficial para a próxima sessão, por empresa: uma ou duas estimações do GARCH em vez do
    backtest inteiro. Usa os últimos `window` + 45 dias (chega para a re-estimação do mês em curso)."""
    rows = []
    for t in tickers:
        r = prices[t].dropna().pct_change().dropna()
        rows.append({"empresa": t} | bt.next_day_forecast(r.tail(window + 45), window=window))
    return pd.DataFrame(rows)


def needs_full(asof, tickers: list[str], anterior: pd.DataFrame | None, weekly: bool = COMPANY_FULL_WEEKLY) -> bool:
    """Backtest completo? Sim se não houver um anterior, se as empresas mudaram, na última sessão da semana,
    ou sempre que COMPANY_FULL_WEEKLY estiver desligado."""
    if anterior is None or set(anterior["empresa"]) != set(tickers) or not weekly:
        return True
    return ultima_sessao_da_semana(pd.Timestamp(asof).date())


def portfolio_forecast(rets: pd.DataFrame, w: pd.Series, window: int = BT_WINDOW) -> dict | None:
    """VaR e ES oficiais (FHS) da carteira para a próxima sessão.

    Aplica o modelo aos retornos da carteira com os pesos atuais (r_p = Σ w_i r_i em cada dia passado), nos dias
    em que todas as posições têm preço. Devolve None se não houver histórico para a janela + 1 mês.
    """
    r = (rets[list(w.index)].dropna() @ w).rename("carteira")
    if len(r) < window + 25:
        return None
    return bt.next_day_forecast(r.tail(window + 45), window=window) | {"dias": len(r)}


def portfolio_backtest(rets: pd.DataFrame, w: pd.Series, window: int = BT_WINDOW) -> dict | None:
    """Backtest da previsão para esta carteira (pesos atuais aplicados ao passado) + previsão para a próxima sessão.

    Devolve None sem histórico para a janela + 3 meses. Senão: previsao (var99, var975, es975, es975_aj, mult_es,
    data_previsao), testes (linhas do método oficial por nível), serie (data, ret, VaR e ES a 97,5%), dias de
    previsão e as marcas validado / es_subestimado.
    """
    r = (rets[list(w.index)].dropna() @ w).rename("carteira")
    if len(r) < window + 60:
        return None
    res, ser, official = bt.run(r.to_frame(), ["carteira"])
    prev = with_flags(official, flags(res)).iloc[0].to_dict()
    of = res[res["metodo"] == OFFICIAL_VAR_METHOD]
    lv = CVAR_LEVEL
    cols = ["data", "ret", f"var_{OFFICIAL_VAR_METHOD}_{lv}", f"es_{OFFICIAL_VAR_METHOD}_{lv}"]
    return {"previsao": prev, "testes": of.reset_index(drop=True), "dias": int(of["dias"].iloc[0]),
            "serie": ser[cols].set_axis(["data", "ret", "var", "es"], axis=1), "dias_precos": len(r)}


def main(full: bool | None = None) -> Path:
    prices = pd.read_parquet(CLEAN / "prices_empresas.parquet")
    tickers = com_historico(prices)
    sem = [t for t in prices.columns if t not in tickers]
    out = FEATURES / "company_backtest.csv"
    anterior = pd.read_csv(out) if out.exists() else None
    asof = prices.index.max()
    if full is None:
        full = needs_full(asof, tickers, anterior)
    FEATURES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if full:
        res, _, official = bt.run(prices.pct_change(), tickers)
        res = res.rename(columns={"setor": "empresa"})
        official = official.rename(columns={"setor": "empresa"})
        res.to_csv(out, index=False, float_format="%.6f")
    else:
        res, official = anterior, daily_forecast(prices, tickers)
    official = with_flags(official, flags(res))
    official.to_csv(FEATURES / "company_risk.csv", index=False, float_format="%.6f")
    modo = "backtest completo" if full else "só previsão (backtest da última sessão semanal)"
    print(f"[backtest empresas] {len(tickers)} empresas em {time.time() - t0:.0f}s, {modo} → company_risk.csv"
          + (f"; sem histórico suficiente ({COMPANY_MIN_DAYS} dias): {sem}" if sem else ""))
    if full:
        print(resumo_metodos(res).to_string(index=False))
    print(f"[backtest empresas] {OFFICIAL_VAR_METHOD} validado a {CVAR_LEVEL:.1%} em {int(official['validado'].sum())} de "
          f"{len(official)}; ES subestimado em {int(official['es_subestimado'].sum())} "
          f"(correção média ×{official.loc[official['mult_es'] > 1, 'mult_es'].mean():.2f})")
    return out


if __name__ == "__main__":
    main(full=True if "--completo" in sys.argv else None)
