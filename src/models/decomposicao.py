"""Decomposição do risco de cada empresa em mercado, setor e específico (Fase C; QRM, secção 3.4: modelos de fatores).

Funções puras; o `main()` no fim é o único ponto com I/O.
  decompose   r_empresa = a + b_m · r_mercado + b_s · r_setor⊥ + e, com r_setor⊥ = parte do setor que o mercado não explica
  table       uma linha por empresa

Como r_mercado e r_setor⊥ não têm correlação na amostra, a variância separa-se sem termos cruzados:
  pct_mercado + pct_setor + pct_especifico = 1.
Retornos simples diários, janela CVAR_YEARS (a mesma convenção da visão conjunta).

Saída: data/features/company_factors.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import CLEAN, CVAR_YEARS, DECOMP_MIN_DAYS, FEATURES, MARKET  # noqa: E402
from src.collect.empresas import load_holdings  # noqa: E402


def decompose(r: pd.Series, mkt: pd.Series, setor: pd.Series, min_days: int = DECOMP_MIN_DAYS) -> dict | None:
    """Betas e partes da variância de `r`. None sem `min_days` dias comuns ou com variância nula."""
    d = pd.concat([r, mkt, setor], axis=1, join="inner").dropna()
    if len(d) < min_days:
        return None
    y, m, s = (d.iloc[:, i].to_numpy() - d.iloc[:, i].mean() for i in range(3))
    if not (y.var() > 0 and m.var() > 0):
        return None
    s_orto = s - (s @ m) / (m @ m) * m
    b_m = (y @ m) / (m @ m)
    b_s = (y @ s_orto) / (s_orto @ s_orto) if s_orto.var() > 0 else 0.0
    p_m, p_s = b_m ** 2 * (m @ m) / (y @ y), b_s ** 2 * (s_orto @ s_orto) / (y @ y)
    return {"beta_mercado": float(b_m), "beta_setor": float(b_s), "pct_mercado": float(p_m), "pct_setor": float(p_s),
            "pct_especifico": float(1 - p_m - p_s), "n": len(d)}


def table(rets: pd.DataFrame, holdings: pd.DataFrame, market: str = MARKET) -> pd.DataFrame:
    """Uma linha por empresa de `holdings` (colunas setor, ticker) com retornos em `rets`."""
    rows = []
    for h in holdings.itertuples():
        if h.ticker in rets and h.setor in rets:
            dec = decompose(rets[h.ticker], rets[market], rets[h.setor])
            if dec:
                rows.append({"empresa": h.ticker, "setor": h.setor} | dec)
    return pd.DataFrame(rows)


def main() -> Path:
    etfs = pd.read_parquet(CLEAN / "prices.parquet")
    emp = pd.read_parquet(CLEAN / "prices_empresas.parquet")
    prices = etfs.join(emp[[c for c in emp.columns if c not in etfs.columns]], how="inner")
    rets = prices.loc[prices.index > prices.index.max() - pd.DateOffset(years=CVAR_YEARS)].pct_change().iloc[1:]
    t = table(rets, load_holdings())
    out = FEATURES / "company_factors.csv"
    t.to_csv(out, index=False, float_format="%.6f")
    med = t[["pct_mercado", "pct_setor", "pct_especifico"]].median()
    print(f"[decomposição] {len(t)} empresas; mediana: mercado {med['pct_mercado']:.0%}, setor {med['pct_setor']:.0%}, "
          f"específico {med['pct_especifico']:.0%}")
    return out


if __name__ == "__main__":
    main()
