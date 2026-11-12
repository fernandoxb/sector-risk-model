"""Carteira do utilizador: empresas, ETFs setoriais e SPY com pesos escolhidos (funções puras).

  parse_carteira / format_carteira   texto do link ("AAPL:30,MSFT:20") ↔ pesos
  normalizar                         pesos positivos que somam 1
  mapa_setores                       ativo → setor (empresa → o seu ETF; ETF → ele próprio; SPY → Mercado)
  exposicao_setorial                 peso da carteira em cada setor
  risco_carteira                     contribuições de Euler (portfolio.risk_contributions) + beta e downside beta
  stress_setorial                    perda aproximada nos cenários históricos, pelo setor de cada posição
  score_carteira                     média dos scores dos setores, ponderada pela exposição
O risco das empresas é histórico e descritivo (sem previsão FHS nem backtest), como no mapa de risco.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (CVAR_LEVEL, MARKET, PANIC_Q, SECTORS, USER_MARKET_LABEL,  # noqa: E402
                             USER_PORTFOLIO_MAX)
from src.models.cockpit import downside_beta  # noqa: E402
from src.models.portfolio import risk_contributions  # noqa: E402


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
