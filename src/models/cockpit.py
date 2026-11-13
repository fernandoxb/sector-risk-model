"""Motor de cálculo do Cockpit de Risco Setorial (app/cockpit).

Funções puras (sem Streamlit) para poderem ser testadas e reutilizadas:
  cvar             perda média nos piores (1 - alpha) dos dias, como número positivo
  downside_beta    cov(ativo, SPY) / var(SPY) só nos dias de pânico do SPY
  tail_table       CVaR + Downside Beta + beta para várias séries
  mst              Minimum Spanning Tree sobre d = sqrt(2(1 - r))
  rate_regime      nível e variação do Treasury a 10 anos (FRED) → regime de juros
  rate_stress      CVaR com um choque de juros, pelo β medido de cada setor (só ligações significativas)
  obsidian_log     nota Markdown diária com links [[Ticker]]
"""
from __future__ import annotations

import io
import os
import urllib.request
from datetime import date

import networkx as nx
import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------------------------------------------------------- dados
def download_close(tickers: list[str], period: str, allow_missing: bool = False) -> pd.DataFrame:
    """Preços ajustados de fecho. Um único pedido, sem threads (como o Bot Coletor).

    Com allow_missing, devolve só os tickers que vieram (em vez de falhar)."""
    df = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=False)
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].set_axis(tickers, axis=1)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    missing = [t for t in tickers if t not in close or close[t].dropna().empty]
    if missing and not allow_missing:
        raise RuntimeError(f"Sem dados do Yahoo para: {missing}")
    return close[[t for t in tickers if t not in missing]]


def returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Retornos simples diários no período comum a todas as séries."""
    return prices.dropna().pct_change().dropna()


def fetch_rates(series: str = "DGS10", start: str | None = None) -> pd.Series:
    """Série diária do FRED em pontos percentuais.

    Usa fredapi se FRED_API_KEY estiver definida; caso contrário o CSV público do FRED.
    """
    key = os.environ.get("FRED_API_KEY")
    if key:
        from fredapi import Fred
        s = Fred(api_key=key).get_series(series, observation_start=start)
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        if start:
            url += f"&cosd={start}"
        import ssl

        import certifi  # o Python do python.org não usa os certificados do macOS
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, timeout=20, context=ctx) as resp:
            raw = pd.read_csv(io.BytesIO(resp.read()), na_values=".")
        s = raw.set_index(raw.columns[0])[series]
    s.index = pd.to_datetime(s.index)
    return pd.to_numeric(s, errors="coerce").dropna().rename(series)


# ----------------------------------------------------------------------------- risco de cauda
def cvar(r: pd.Series, alpha: float) -> float:
    """Expected Shortfall histórico: média dos retornos <= quantil (1 - alpha), em perda positiva."""
    cut = r.quantile(1 - alpha)
    return float(-r[r <= cut].mean())


def downside_beta(r: pd.Series, mkt: pd.Series, panic_q: float) -> float:
    """Beta condicional: regressão do ativo no mercado apenas nos dias em que o mercado
    ficou no percentil `panic_q` inferior."""
    both = pd.concat([r, mkt], axis=1, join="inner").dropna()
    x, m = both.iloc[:, 0], both.iloc[:, 1]
    panic = m <= m.quantile(panic_q)
    if panic.sum() < 10:
        return float("nan")
    xm, mm = x[panic], m[panic]
    var_m = mm.var()
    return float(np.cov(xm, mm)[0, 1] / var_m) if var_m > 0 else float("nan")


def tail_table(rets: pd.DataFrame, mkt: pd.Series, alpha: float, panic_q: float) -> pd.DataFrame:
    rows = {}
    for t in rets.columns:
        rows[t] = {
            "cvar": cvar(rets[t], alpha),
            "downside_beta": downside_beta(rets[t], mkt, panic_q),
            "beta": float(np.cov(rets[t], mkt.reindex(rets.index))[0, 1] / mkt.reindex(rets.index).var()),
        }
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------- rede
def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    return np.sqrt((2 * (1 - corr)).clip(lower=0))


def mst(rets: pd.DataFrame) -> tuple[nx.Graph, pd.DataFrame]:
    corr = rets.corr()
    dist = correlation_distance(corr)
    g = nx.Graph()
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            g.add_edge(a, b, weight=float(dist.loc[a, b]), corr=float(corr.loc[a, b]))
    return nx.minimum_spanning_tree(g, weight="weight"), corr


# ----------------------------------------------------------------------------- regime de juros
def rate_regime(y: pd.Series, lookback: int = 63, threshold_pp: float = 0.25) -> dict:  # padrões = settings
    """Nível atual e variação em `lookback` dias úteis do Treasury 10Y."""
    y = y.dropna()
    level, past = float(y.iloc[-1]), float(y.iloc[-1 - lookback])
    delta = level - past
    regime = "subida" if delta >= threshold_pp else "descida" if delta <= -threshold_pp else "estável"
    return {"asof": str(y.index[-1].date()), "nivel": level, "delta": delta, "regime": regime}


def rate_stress(cvar_s: pd.Series, betas: pd.Series, delta_pp: float) -> pd.Series:
    """CVaR com um choque de juros por cima: CVaR_i + max(0, −(e^{b_i·Δ} − 1)).

    `betas` são os β medidos (retorno log semanal do setor por +1 p.p. no Treasury 10Y, regressão HAC), só das
    ligações aceites; setores sem ligação significativa (NaN ou ausentes) não mudam. A perda do choque só soma
    quando é perda: um setor que ganha com o choque fica com o CVaR original.
    """
    b = betas.reindex(cvar_s.index).fillna(0.0)
    perda = -np.expm1(b * delta_pp)
    return cvar_s + perda.clip(lower=0.0)


# ----------------------------------------------------------------------------- Obsidian
def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def obsidian_log(day: date, params: dict, sectors: pd.DataFrame, edges: list[tuple[str, str, float]],
                 regime: dict | None, drill_sector: str | None = None,
                 drill: pd.DataFrame | None = None) -> str:
    """Nota diária. `sectors` e `drill` já vêm formatados; a 1.ª coluna é o ticker."""
    sectors = sectors.copy()
    sectors.iloc[:, 0] = [f"[[{t}]]" for t in sectors.iloc[:, 0]]
    fm = {"tipo": "log-risco-setorial", "data": day.isoformat(), **params}
    if regime:
        fm |= {"dgs10": round(regime["nivel"], 2), "delta_10y_pp": round(regime["delta"], 2),
               "regime_juros": regime["regime"]}
    lines = ["---", *[f"{k}: {v}" for k, v in fm.items()], "---", "",
             f"# Risco Setorial — {day.isoformat()}", "", "## Setores", "", _md_table(sectors), "",
             "## Espinha de contágio (MST)", ""]
    lines += [f"- [[{a}]] ↔ [[{b}]] — r = {c:.2f}" for a, b, c in edges]
    if drill_sector and drill is not None and not drill.empty:
        drill = drill.copy()
        drill.iloc[:, 0] = [f"[[{t}]]" for t in drill.iloc[:, 0]]
        lines += ["", f"## Microscópio: [[{drill_sector}]]", "", _md_table(drill)]
    return "\n".join(lines) + "\n"
