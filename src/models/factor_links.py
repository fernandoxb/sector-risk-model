"""Ligações medidas entre fatores macro e setores (Sprint 4c).

Funções puras (sem Streamlit, sem I/O) — o `main()` no fim é o único ponto com I/O:
  weekly_cut          data de corte: a última sexta-feira com semana completa
  weekly_levels       nível no fim de cada semana (sexta a sexta)
  factor_changes      variação semanal de cada fator (Δ p.p. ou Δ log, conforme MACRO_FACTORS)
  sector_returns      retornos log semanais dos setores
  hac_lags            nº de lags de Newey–West: floor(4 · (T/100)^(2/9))
  regress             OLS r_setor = a + b · Δfator com erros-padrão HAC (Newey–West)
  link_matrix         todas as regressões fator × setor, com sinal, aceitação e intensidade
  contagion_links     arestas da MST entre setores (`cockpit.mst`) como ligações de contágio
  links_payload       lista de ligações para o risk_map_data.json

Sinal (a mesma convenção das ligações manuais do mapa):
  "+"  o movimento adverso do fator piora o setor   (adverso · beta < 0)
  "−"  o movimento adverso do fator melhora o setor (adverso · beta > 0)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (CLEAN, CVAR_YEARS, FEATURES, LINK_FREQ, LINK_HAC_LAGS,  # noqa: E402
                             LINK_T_MIN, MACRO_FACTORS, SECTORS)
from src.models import cockpit as ck  # noqa: E402

PLUS, MINUS = "+", "-"


# ----------------------------------------------------------------------------- dados semanais
def weekly_cut(last_day: pd.Timestamp) -> pd.Timestamp:
    """Última sexta-feira <= last_day (a semana em curso, incompleta, fica de fora)."""
    return last_day - pd.Timedelta(days=(last_day.weekday() - 4) % 7)


def weekly_levels(df: pd.DataFrame, freq: str = LINK_FREQ) -> pd.DataFrame:
    """Último valor disponível de cada semana (rótulo = sexta-feira)."""
    return df.resample(freq).last()


def factor_changes(levels: pd.DataFrame, factors: dict = MACRO_FACTORS) -> pd.DataFrame:
    """Δ em p.p. ("diff") ou Δ log ("logdiff"; níveis <= 0 ficam NaN, ex.: WTI a 2020-04-20)."""
    out = {}
    for k, f in factors.items():
        x = levels[k]
        if f["transf"] == "diff":
            out[k] = x.diff()
        elif f["transf"] == "logdiff":
            out[k] = np.log(x.where(x > 0)).diff()
        else:
            raise ValueError(f"Transformação desconhecida: {f['transf']}")
    return pd.DataFrame(out)


def sector_returns(weekly_prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(weekly_prices).diff()


# ----------------------------------------------------------------------------- regressão
def hac_lags(n: int) -> int:
    """Regra de Newey–West (1994): floor(4 · (T/100)^(2/9))."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def regress(y: pd.Series, x: pd.Series, lags: int | None = LINK_HAC_LAGS) -> dict:
    """OLS de y em x com constante; t e p com erros-padrão HAC (núcleo de Bartlett)."""
    d = pd.concat([y, x], axis=1, join="inner").dropna()
    yy, xx = d.iloc[:, 0], d.iloc[:, 1]
    n = len(d)
    L = hac_lags(n) if lags is None else lags
    res = sm.OLS(yy.to_numpy(), sm.add_constant(xx.to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    beta = float(res.params[1])
    return {"beta": beta, "se": float(res.bse[1]), "t": float(res.tvalues[1]), "p": float(res.pvalues[1]),
            "r2": float(res.rsquared), "n": n, "lags": L,
            "beta_pad": beta * float(xx.std()) / float(yy.std())}   # = correlação na regressão simples


def tercile(s: pd.Series) -> pd.Series:
    """Intensidade 1/2/3 pelos tercis (percentil da posição entre os valores dados)."""
    return np.ceil(s.rank(pct=True) * 3).clip(1, 3).astype(int)


def link_matrix(rets: pd.DataFrame, changes: pd.DataFrame, factors: dict = MACRO_FACTORS,
                t_min: float = LINK_T_MIN, lags: int | None = LINK_HAC_LAGS) -> pd.DataFrame:
    """Uma linha por fator × setor. `aceite` se |t| >= t_min; `w` (1–3) só nas aceites."""
    rows = []
    for k, f in factors.items():
        for s in rets.columns:
            r = regress(rets[s], changes[k], lags)
            aceite = abs(r["t"]) >= t_min
            rows.append({"fator": k, "setor": s, **r, "aceite": aceite,
                         "sinal": (PLUS if f["adverso"] * r["beta"] < 0 else MINUS) if aceite else None})
    m = pd.DataFrame(rows)
    m["w"] = pd.NA
    acc = m["aceite"]
    if acc.any():
        m.loc[acc, "w"] = tercile(m.loc[acc, "beta_pad"].abs()).to_numpy()
    return m


def contagion_links(daily: pd.DataFrame) -> pd.DataFrame:
    """Arestas da MST (`cockpit.mst`, a mesma do cockpit) com a correlação e intensidade por tercil."""
    tree, _ = ck.mst(daily)
    e = pd.DataFrame([{"a": a, "b": b, "corr": d["corr"]} for a, b, d in tree.edges(data=True)])
    e["w"] = tercile(e["corr"])
    return e


def links_payload(m: pd.DataFrame, cont: pd.DataFrame) -> list[dict]:
    """Ligações no formato do mapa: {from, to, sign, w, beta, t, r2, tipo} (ids em minúsculas)."""
    out = [{"from": r.fator, "to": r.setor.lower(), "sign": r.sinal, "w": int(r.w),
            "beta": round(r.beta, 6), "t": round(r.t, 3), "r2": round(r.r2, 4), "n": int(r.n),
            "tipo": "medida"} for r in m[m["aceite"]].itertuples()]
    out += [{"from": r.a.lower(), "to": r.b.lower(), "sign": None, "w": int(r.w), "beta": None, "t": None,
             "r2": None, "corr": round(r.corr, 4), "tipo": "contágio"} for r in cont.itertuples()]
    return out


# ----------------------------------------------------------------------------- I/O
def main() -> Path:
    prices = pd.read_parquet(CLEAN / "prices.parquet")[list(SECTORS)]
    factors = pd.read_parquet(CLEAN / "factors.parquet")
    end = weekly_cut(prices.index.max())
    start = prices.index.max() - pd.DateOffset(years=CVAR_YEARS)
    rets = sector_returns(weekly_levels(prices.loc[:end]))
    chg = factor_changes(weekly_levels(factors.loc[:end]))
    rets, chg = rets.loc[rets.index > start], chg.loc[chg.index > start]

    m = link_matrix(rets, chg)
    daily = prices.loc[prices.index > start].pct_change().dropna()   # MST com retornos diários, como o cockpit
    cont = contagion_links(daily)

    out = FEATURES / "factor_links.csv"
    m.to_csv(out, index=False, float_format="%.6f")
    mj = FEATURES / "risk_map_data.json"
    payload = json.loads(mj.read_text())
    payload["links"] = links_payload(m, cont)
    payload["links_metodo"] = {
        "frequencia": "semanal (sexta a sexta)", "inicio": str(rets.index[0].date()), "fim": str(end.date()),
        "t_min": LINK_T_MIN, "erros_padrao": "HAC Newey–West", "lags": int(m["lags"].iloc[0]),
        "sinal": "+ = o movimento adverso do fator piora o setor",
        "fatores": {k: {kk: v for kk, v in f.items()} for k, f in MACRO_FACTORS.items()},
    }
    mj.write_text(json.dumps(payload, ensure_ascii=False, indent=1))

    print(f"[ligações] {int(m['aceite'].sum())}/{len(m)} ligações fator → setor aceites (|t| ≥ {LINK_T_MIN}), "
          f"{len(cont)} de contágio; semanas {rets.index[0].date()} → {end.date()} "
          f"(n = {int(m['n'].min())}–{int(m['n'].max())}, {int(m['lags'].iloc[0])} lags HAC)")
    print(m.pivot(index="setor", columns="fator", values="t").round(2).to_string())
    return out


if __name__ == "__main__":
    main()
