"""Ligações medidas entre fatores macro e setores (Sprint 4c).

Funções puras (sem Streamlit, sem I/O) — o `main()` no fim é o único ponto com I/O:
  weekly_cut          data de corte: a última sexta-feira com semana completa
  weekly_levels       nível no fim de cada semana (sexta a sexta)
  factor_changes      variação semanal de cada fator (Δ p.p. ou Δ log, conforme MACRO_FACTORS)
  sector_returns      retornos log semanais dos setores
  hac_lags            nº de lags de Newey–West: floor(4 · (T/100)^(2/9))
  regress             OLS r_setor = a + b · Δfator com erros-padrão HAC (Newey–West)
  link_matrix         todas as regressões fator × setor, com sinal, aceitação e intensidade
  fit_multivariate_ridge  os fatores todos ao mesmo tempo (Ridge em variáveis padronizadas + t HAC conjunto)

Sinal (a mesma convenção das ligações manuais do mapa):
  "+"  o movimento adverso do fator piora o setor   (adverso · beta < 0)
  "−"  o movimento adverso do fator melhora o setor (adverso · beta > 0)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (CLEAN, CVAR_YEARS, FACTOR_BETA_MIN, FACTOR_RIDGE_ALPHA, FEATURES,  # noqa: E402
                             LINK_FREQ, LINK_HAC_LAGS, LINK_T_MIN, MACRO_FACTORS, SECTORS)

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


def fit_multivariate_ridge(rets: pd.DataFrame, changes: pd.DataFrame, alpha: float = FACTOR_RIDGE_ALPHA,
                           beta_min: float = FACTOR_BETA_MIN, factors: dict = MACRO_FACTORS,
                           t_min: float = LINK_T_MIN, lags: int | None = LINK_HAC_LAGS) -> pd.DataFrame:
    """Regressão conjunta de cada setor em todos os fatores ao mesmo tempo. Uma linha por fator × setor.

    Só entram as semanas em que todos os fatores e setores têm valor (o FRED pode vir uma semana atrás do Yahoo).
    Fatores e retornos são padronizados (média 0, desvio 1), por isso o coeficiente do Ridge é o `beta_pad_multi`,
    comparável entre fatores com unidades diferentes: β = (Z'Z + αI)⁻¹ Z'y. `b_multi` é o mesmo coeficiente na
    escala original (β_pad · σ_y / σ_x). `t_multi` é o t HAC da regressão conjunta sem penalização.
    `aceite_multi`: |beta_pad_multi| >= beta_min e |t_multi| >= t_min; `sinal_multi` com a convenção de `link_matrix`.
    """
    k = list(factors)
    d = rets.join(changes[k], how="inner").dropna()
    X, Y = d[k], d[list(rets.columns)]
    sx, sy = X.std(), Y.std()
    zx, zy = ((X - X.mean()) / sx).to_numpy(), ((Y - Y.mean()) / sy).to_numpy()
    B = np.linalg.solve(zx.T @ zx + alpha * np.eye(len(k)), zx.T @ zy)          # fatores × setores
    L = hac_lags(len(d)) if lags is None else lags
    rows = []
    for j, s in enumerate(Y.columns):
        t = sm.OLS(Y[s].to_numpy(), sm.add_constant(X.to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": L}).tvalues[1:]
        for i, f in enumerate(k):
            bp = float(B[i, j])
            aceite = abs(bp) >= beta_min and abs(t[i]) >= t_min
            rows.append({"fator": f, "setor": s, "beta_pad_multi": bp, "b_multi": bp * sy[s] / sx[f],
                         "t_multi": float(t[i]), "n_multi": len(d), "aceite_multi": aceite,
                         "sinal_multi": (PLUS if factors[f]["adverso"] * bp < 0 else MINUS) if aceite else None})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- I/O
def main() -> Path:
    prices = pd.read_parquet(CLEAN / "prices.parquet")[list(SECTORS)]
    factors = pd.read_parquet(CLEAN / "factors.parquet")
    end = weekly_cut(prices.index.max())
    start = prices.index.max() - pd.DateOffset(years=CVAR_YEARS)
    rets = sector_returns(weekly_levels(prices.loc[:end]))
    chg = factor_changes(weekly_levels(factors.loc[:end]))
    rets, chg = rets.loc[rets.index > start], chg.loc[chg.index > start]

    m = link_matrix(rets, chg).merge(fit_multivariate_ridge(rets, chg), on=["fator", "setor"], how="left")

    out = FEATURES / "factor_links.csv"
    m.to_csv(out, index=False, float_format="%.6f")

    print(f"[ligações] {int(m['aceite'].sum())}/{len(m)} ligações fator → setor aceites (|t| ≥ {LINK_T_MIN}), "
          f"semanas {rets.index[0].date()} → {end.date()} "
          f"(n = {int(m['n'].min())}–{int(m['n'].max())}, {int(m['lags'].iloc[0])} lags HAC)")
    print(m.pivot(index="setor", columns="fator", values="t").round(2).to_string())
    print(f"[ligações] regressão conjunta: {int(m['aceite_multi'].sum())} resistem aos outros fatores; "
          f"{int((m['aceite'] & ~m['aceite_multi'].astype(bool)).sum())} das aceites uma a uma deixam de o ser")
    return out


if __name__ == "__main__":
    main()
