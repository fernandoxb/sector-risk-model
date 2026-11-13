"""Carteira do utilizador: empresas, ETFs setoriais e SPY com pesos escolhidos (funções puras).

  parse_carteira / format_carteira   texto do link ("AAPL:30,MSFT:20") ↔ pesos
  normalizar                         pesos positivos que somam 1
  mapa_setores                       ativo → setor (empresa → o seu ETF; ETF → ele próprio; SPY → Mercado)
  exposicao_setorial                 peso da carteira em cada setor
  risco_carteira                     contribuições de Euler (portfolio.risk_contributions) + beta e downside beta
  stress_setorial                    perda aproximada nos cenários históricos, pelo setor de cada posição
  score_carteira                     média dos scores dos setores, ponderada pela exposição
  calcular_pesos_hrp                 pesos por Hierarchical Risk Parity, agrupando pelos ativos que caem juntos
  hrp_fora_da_amostra, resumo_oos    a HRP recalculada só com o passado, contra os pesos do utilizador e pesos iguais
O risco das empresas é histórico e descritivo (sem previsão FHS nem backtest), como no mapa de risco.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

from config.settings import (CVAR_LEVEL, HRP_LINKAGE, HRP_MIN_DAYS, HRP_OOS_JANELA, HRP_OOS_PASSO,  # noqa: E402
                             HRP_RISK_MEASURE, MARKET, PANIC_Q, TRADING_DAYS,
                             SECTORS, USER_MARKET_LABEL, USER_PORTFOLIO_MAX)
from src.models.cauda import joint_crash  # noqa: E402
from src.models.cockpit import cvar, downside_beta  # noqa: E402
from src.models.portfolio import ledoit_wolf, risk_contributions  # noqa: E402


def parse_carteira(texto: str | None, universo: set[str] | None = None) -> dict[str, float]:
    """"AAPL:30, msft:20;XLU:5%" → {"AAPL": 30.0, "MSFT": 20.0, "XLU": 5.0}.

    Ignora entradas mal formadas, pesos <= 0 e ativos fora do `universo`; repetidos somam; no máximo
    USER_PORTFOLIO_MAX posições."""
    out: dict[str, float] = {}
    for parte in (texto or "").replace(";", ",").split(","):
        if ":" not in parte:
            continue
        t, _, p = parte.partition(":")
        t = t.strip().upper()
        try:
            v = float(p.strip().rstrip("%").replace(",", "."))
        except ValueError:
            continue
        if not t or v <= 0 or not np.isfinite(v) or (universo is not None and t not in universo):
            continue
        if t in out or len(out) < USER_PORTFOLIO_MAX:
            out[t] = out.get(t, 0.0) + v
    return out


def format_carteira(pesos: dict[str, float]) -> str:
    """{"AAPL": 0.3, "MSFT": 0.2} (qualquer escala) → "AAPL:60,MSFT:40" (percentagens, 1 casa no máximo)."""
    w = normalizar(pesos) * 100
    return ",".join(f"{t}:{v:.1f}".rstrip("0").rstrip(".") for t, v in w.items())


def normalizar(pesos: dict[str, float]) -> pd.Series:
    """Pesos positivos a somar 1, pela ordem dada. ValueError se não houver nenhum."""
    w = pd.Series({t: float(v) for t, v in pesos.items() if v and float(v) > 0}, dtype=float, name="peso")
    if w.empty:
        raise ValueError("A carteira não tem nenhuma posição com peso positivo")
    return w / w.sum()


def mapa_setores(holdings: pd.DataFrame) -> dict[str, str]:
    """Ativo → setor. Uma empresa presente em dois ETFs fica no de maior peso."""
    h = holdings.sort_values("peso", ascending=False).drop_duplicates("ticker")
    return dict(zip(h["ticker"], h["setor"])) | {s: s for s in SECTORS} | {MARKET: USER_MARKET_LABEL}


def exposicao_setorial(w: pd.Series, setores: dict[str, str]) -> pd.Series:
    """Peso da carteira em cada um dos 11 setores (0 se não tiver) e, se houver SPY, em "Mercado"."""
    e = w.groupby(w.index.map(setores)).sum()
    idx = list(SECTORS) + [s for s in e.index if s not in SECTORS]
    return e.reindex(idx).fillna(0.0).rename("exposicao")


def risco_carteira(rets: pd.DataFrame, w: pd.Series, mkt: pd.Series, alpha: float = CVAR_LEVEL,
                   panic_q: float = PANIC_Q) -> tuple[pd.DataFrame, dict]:
    """Tabela por posição (peso, risco isolado, contribuição para a volatilidade e o CVaR, beta) e resumo da
    carteira (volatilidade, CVaR, diversificação, beta e downside beta face ao mercado)."""
    tab, resumo = risk_contributions(rets, w, alpha)
    r = rets[w.index].dropna()
    m = mkt.reindex(r.index)
    tab["beta"] = [float(np.cov(r[c], m)[0, 1] / m.var()) for c in r.columns]
    tab["racio"] = tab["pct_cvar"] / tab["peso"]              # > 1: pesa mais no risco do que no capital
    rp = r @ w
    resumo |= {"beta": float(np.cov(rp, m)[0, 1] / m.var()), "downside_beta": downside_beta(rp, m, panic_q),
               "posicoes": len(w), "maior_peso": float(w.max())}
    return tab, resumo


def stress_setorial(expo: pd.Series, stress: pd.DataFrame) -> pd.DataFrame:
    """Perda aproximada da carteira em cada cenário histórico: cada posição perde o que perdeu o seu setor
    (o SPY, a perda do SPY). Ignora o risco específico das empresas."""
    piv = stress.pivot(index="serie", columns="cenario", values="perda")
    piv.loc[USER_MARKET_LABEL] = piv.loc[MARKET]
    e = expo[expo > 0]
    perda = piv.reindex(e.index).mul(e, axis=0).sum()
    pior = piv.reindex(e.index).mul(e, axis=0).idxmax()
    return pd.DataFrame({"perda": perda, "perda_mercado": piv.loc[MARKET], "maior_contribuinte": pior}).rename_axis(
        "cenario").reset_index()


def score_carteira(expo: pd.Series, scores: pd.Series) -> float:
    """Média dos scores dos setores ponderada pela exposição (só a parte investida em setores)."""
    e = expo.reindex(scores.index).fillna(0.0)
    return float((e * scores).sum() / e.sum()) if e.sum() > 0 else float("nan")


# ----------------------------------------------------------------------------- Hierarchical Risk Parity
def hrp_quasi_diag(link: np.ndarray) -> list[int]:
    """Ordem das folhas da árvore: ativos que caem juntos ficam lado a lado (López de Prado, 2016, passo 2)."""
    return [int(i) for i in leaves_list(link)]


def _risco_ramo(rets: pd.DataFrame, cov: pd.DataFrame, ativos: list[str], risk_measure: str) -> float:
    """Risco de um ramo, com pesos inversos à variância dentro dele: CVaR histórico ou variância (Ledoit-Wolf)."""
    iv = 1 / np.diag(cov.loc[ativos, ativos])
    w = iv / iv.sum()
    if risk_measure == "variance":
        return float(w @ cov.loc[ativos, ativos].to_numpy() @ w)
    if risk_measure == "cvar":
        return max(cvar(rets[ativos] @ w, CVAR_LEVEL), 1e-12)        # um ramo sem perdas na cauda não fica com tudo
    raise ValueError(f"Medida de risco desconhecida: {risk_measure}")


def hrp_recursive_bisection(rets: pd.DataFrame, ordem_ativos: list[str],
                            risk_measure: str = HRP_RISK_MEASURE) -> pd.Series:
    """Divide a lista ordenada ao meio, vezes sem conta; em cada corte, o ramo com menos risco recebe mais peso:
    α₁ = 1 − R₁ / (R₁ + R₂), α₂ = 1 − α₁ (López de Prado, 2016, passo 3)."""
    cov = pd.DataFrame(ledoit_wolf(rets[ordem_ativos].to_numpy())[0], index=ordem_ativos, columns=ordem_ativos)
    w = pd.Series(1.0, index=ordem_ativos, name="peso")
    ramos = [list(ordem_ativos)]
    while ramos:
        ramos = [metade for r in ramos if len(r) > 1 for metade in (r[:len(r) // 2], r[len(r) // 2:])]
        for c1, c2 in zip(ramos[::2], ramos[1::2]):
            r1, r2 = _risco_ramo(rets, cov, c1, risk_measure), _risco_ramo(rets, cov, c2, risk_measure)
            a1 = 1 - r1 / (r1 + r2)
            w[c1] *= a1
            w[c2] *= 1 - a1
    return w / w.sum()


def calcular_pesos_hrp(retornos_ativos: pd.DataFrame, matriz_distancia: pd.DataFrame | None = None,
                       risk_measure: str = HRP_RISK_MEASURE, min_days: int = HRP_MIN_DAYS) -> pd.Series | None:
    """Pesos HRP (somam 1, todos >= 0), pela ordem das colunas. None se houver menos de `min_days` dias comuns.

    Sem `matriz_distancia`, usa d = 1 − p_conj (`cauda.joint_crash`: frequência com que dois ativos estão juntos
    nos piores dias), calculada nos próprios retornos, porque a carteira pode misturar empresas, ETFs e SPY."""
    r = retornos_ativos.dropna()
    cols = list(r.columns)
    if len(cols) == 1:
        return pd.Series({cols[0]: 1.0}, name="peso")
    if len(r) < min_days:
        return None
    d = (1 - joint_crash(r)) if matriz_distancia is None else matriz_distancia.loc[cols, cols]
    d = d.to_numpy(dtype=float)
    d = np.nan_to_num((d + d.T) / 2, nan=1.0)                 # simétrica; par sem dados = sem ligação
    np.fill_diagonal(d, 0.0)
    ordem = [cols[i] for i in hrp_quasi_diag(linkage(squareform(d, checks=False), HRP_LINKAGE))]
    return hrp_recursive_bisection(r, ordem, risk_measure).reindex(cols)


def hrp_fora_da_amostra(rets: pd.DataFrame, w: pd.Series, janela: int = HRP_OOS_JANELA, passo: int = HRP_OOS_PASSO,
                        risk_measure: str = HRP_RISK_MEASURE) -> pd.DataFrame | None:
    """Retornos diários fora da amostra de três carteiras com os mesmos ativos: a HRP, recalculada a cada `passo` dias
    só com os `janela` dias anteriores; os pesos `w` do utilizador; e pesos iguais. Pesos constantes dentro de cada
    período. None se não houver `janela` + `passo` dias em que todos os ativos têm preço."""
    r = rets[list(w.index)].dropna()
    if len(r) < janela + passo:
        return None
    partes = []
    for i in range(janela, len(r), passo):
        w_h = calcular_pesos_hrp(r.iloc[i - janela:i], risk_measure=risk_measure, min_days=min(HRP_MIN_DAYS, janela))
        bloco = r.iloc[i:i + passo]
        partes.append(pd.DataFrame({"hrp": bloco @ w_h, "atual": bloco @ w, "iguais": bloco.mean(axis=1)}))
    return pd.concat(partes)


def resumo_oos(oos: pd.DataFrame, alpha: float = CVAR_LEVEL) -> pd.DataFrame:
    """Por carteira: volatilidade anual e CVaR a 1 dia dos retornos fora da amostra."""
    return pd.DataFrame({"vol": oos.std() * np.sqrt(TRADING_DAYS), "cvar": oos.apply(lambda c: cvar(c, alpha))})

