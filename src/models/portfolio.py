"""Visão conjunta dos 11 setores (Sprint 3).

Funções puras (sem Streamlit, sem I/O) — o `main()` no fim é o único ponto com I/O:
  portfolio_weights    pesos da carteira de referência (iguais por omissão)
  ledoit_wolf          covariância encolhida para a identidade escalada (Ledoit & Wolf, 2004)
  shrinkage_table      Ledoit-Wolf vs amostral: intensidade, nº de condição, variância mínima fora da amostra
  risk_contributions   decomposição de Euler da volatilidade e do CVaR da carteira por setor
  stress_test          perdas por setor e da carteira nos episódios históricos (pico → vale do SPY)

Convenções: a carteira agrega-se em **retornos simples** (r_p = Σ w_i r_i só é exato em simples),
por isso aqui também a volatilidade usa retornos simples. Perdas em número positivo.
O CVaR é sempre `src/models/cockpit.py::cvar`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (CLEAN, CVAR_LEVEL, CVAR_YEARS, FEATURES, LW_REBAL_FREQ,  # noqa: E402
                             LW_WINDOWS, MARKET, PORTFOLIO_WEIGHTS, SECTORS, STRESS_SCENARIOS,
                             TRADING_DAYS)
from src.models.backtest import refit_positions  # noqa: E402
from src.models.cockpit import cvar  # noqa: E402
from src.models.risk_metrics import drawdown  # noqa: E402

ANN = np.sqrt(TRADING_DAYS)


# ----------------------------------------------------------------------------- carteira
def portfolio_weights(tickers: list[str], custom: dict[str, float] | None = PORTFOLIO_WEIGHTS) -> pd.Series:
    """Pesos iguais, ou os de `custom` (têm de cobrir os `tickers` e somar 1)."""
    if custom is None:
        return pd.Series(1 / len(tickers), index=tickers, name="peso")
    w = pd.Series(custom, name="peso").reindex(tickers)
    if w.isna().any() or not np.isclose(w.sum(), 1):
        raise ValueError("PORTFOLIO_WEIGHTS tem de ter um peso para cada setor e somar 1")
    return w


# ----------------------------------------------------------------------------- Ledoit-Wolf
def sample_cov(x: np.ndarray) -> np.ndarray:
    """Covariância amostral de máxima verosimilhança (divide por T), a base do Ledoit-Wolf."""
    xc = x - x.mean(axis=0)
    return xc.T @ xc / len(xc)


def ledoit_wolf(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Σ_LW = δ·m·I + (1 - δ)·S, com m = tr(S)/N e δ ∈ [0, 1] estimado (Ledoit & Wolf, 2004).

    Com ‖A‖² = tr(AAᵀ)/N:
        d² = ‖S - mI‖²,  b̄² = (1/T²) Σ_t ‖x_t x_tᵀ - S‖²,  b² = min(b̄², d²),  δ = b²/d²
    """
    x = np.asarray(x, dtype=float)
    t, n = x.shape
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / t
    m = np.trace(s) / n
    d2 = ((s - m * np.eye(n)) ** 2).sum() / n
    # Σ_t ‖x_t x_tᵀ - S‖_F² = Σ_t (x_tᵀx_t)² - T‖S‖_F²   (porque Σ_t x_t x_tᵀ = T·S)
    b2bar = (((xc ** 2).sum(axis=1) ** 2).sum() - t * (s ** 2).sum()) / (n * t ** 2)
    delta = float(min(b2bar, d2) / d2) if d2 > 0 else 1.0
    return delta * m * np.eye(n) + (1 - delta) * s, delta


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Carteira de variância mínima sem restrições: w = Σ⁻¹1 / 1ᵀΣ⁻¹1."""
    a = np.linalg.solve(cov, np.ones(len(cov)))
    return a / a.sum()


def oos_min_variance(rets: pd.DataFrame, window: int, freq: str = LW_REBAL_FREQ) -> pd.DataFrame:
    """Retornos diários fora da amostra da carteira de variância mínima, estimada com a covariância
    amostral e com a Ledoit-Wolf nos `window` dias anteriores e re-balanceada em cada período `freq`."""
    x = rets.to_numpy()
    pos = refit_positions(rets.index, window, freq)
    out = np.full((len(x), 2), np.nan)
    for k, i in enumerate(pos):
        end = pos[k + 1] if k + 1 < len(pos) else len(x)
        hist = x[i - window:i]
        for j, cov in enumerate((sample_cov(hist), ledoit_wolf(hist)[0])):
            out[i:end, j] = x[i:end] @ min_variance_weights(cov)
    return pd.DataFrame(out, index=rets.index, columns=["amostral", "ledoit_wolf"]).dropna()


def shrinkage_table(rets: pd.DataFrame, windows: tuple[int, ...] = LW_WINDOWS,
                    freq: str = LW_REBAL_FREQ) -> pd.DataFrame:
    """Por janela: δ e nº de condição na janela mais recente + volatilidade realizada (anualizada)
    da variância mínima fora da amostra com cada estimador."""
    rows = []
    for win in windows:
        last = rets.iloc[-win:].to_numpy()
        s = sample_cov(last)
        lw, delta = ledoit_wolf(last)
        oos = oos_min_variance(rets, win, freq)
        vol = oos.std() * ANN
        rows.append({
            "janela": win, "t_sobre_n": win / rets.shape[1], "delta": delta,
            "cond_amostral": float(np.linalg.cond(s)), "cond_ledoit_wolf": float(np.linalg.cond(lw)),
            "oos_inicio": str(oos.index[0].date()), "oos_dias": len(oos),
            "vol_oos_amostral": float(vol["amostral"]), "vol_oos_ledoit_wolf": float(vol["ledoit_wolf"]),
            "reducao_vol": float(1 - vol["ledoit_wolf"] / vol["amostral"]),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- contribuição para o risco
def risk_contributions(rets: pd.DataFrame, w: pd.Series, alpha: float = CVAR_LEVEL) -> tuple[pd.DataFrame, dict]:
    """Decomposição de Euler (as contribuições somam o risco da carteira).

    Volatilidade (Σ Ledoit-Wolf):  RC_i = w_i (Σw)_i / σ_p
    CVaR histórico:                RC_i = -w_i · média(r_i nos dias em que r_p <= quantil 1-α)
    """
    rets = rets[w.index].dropna()
    x, wv = rets.to_numpy(), w.to_numpy()
    cov = ledoit_wolf(x)[0]
    sig_p = float(np.sqrt(wv @ cov @ wv))
    rc_vol = wv * (cov @ wv) / sig_p
    rp = pd.Series(x @ wv, index=rets.index)
    es_p = cvar(rp, alpha)
    tail = rp <= rp.quantile(1 - alpha)                   # os mesmos dias que o cvar usa
    rc_es = -wv * x[tail.to_numpy()].mean(axis=0)
    tab = pd.DataFrame({
        "peso": wv,
        "vol_individual": np.sqrt(np.diag(cov)) * ANN,
        "contrib_vol": rc_vol * ANN, "pct_vol": rc_vol / sig_p,
        "cvar_individual": [cvar(rets[c], alpha) for c in rets.columns],
        "contrib_cvar": rc_es, "pct_cvar": rc_es / es_p,
    }, index=rets.columns)
    resumo = {
        "inicio": str(rets.index[0].date()), "fim": str(rets.index[-1].date()), "dias": len(rets),
        "vol": sig_p * ANN, "cvar": es_p, "dias_cauda": int(tail.sum()),
        "racio_diversificacao_vol": float((tab["peso"] * tab["vol_individual"]).sum() / (sig_p * ANN)),
        "racio_diversificacao_cvar": float((tab["peso"] * tab["cvar_individual"]).sum() / es_p),
    }
    return tab, resumo


# ----------------------------------------------------------------------------- stress tests
def _path_stats(r: pd.Series) -> dict:
    """Perda acumulada, pior dia e drawdown máximo de um caminho de retornos (perdas positivas)."""
    value = pd.concat([pd.Series([1.0]), (1 + r).cumprod().reset_index(drop=True)])
    return {"perda": float(1 - value.iloc[-1]), "pior_dia": float(-r.min()),
            "max_dd": float(-drawdown(value).min())}


def stress_test(simple: pd.DataFrame, w: pd.Series, betas: pd.Series,
                scenarios: dict[str, tuple[str, str]] = STRESS_SCENARIOS,
                market: str = MARKET, cvar_now: float | None = None) -> pd.DataFrame:
    """Perdas nos episódios históricos para cada setor, a carteira (buy-and-hold) e o SPY.

    Um setor sem história completa no episódio (XLRE e XLC em 2008) usa o proxy r_i = β_i · r_SPY,
    com o β atual; a coluna `proxy` assinala-o. A contribuição w_i · perda_i soma a perda da carteira.
    """
    rows = []
    for nome, (ini, fim) in scenarios.items():
        win = simple.loc[(simple.index > ini) & (simple.index <= fim)]
        paths, proxy = {}, {}
        for t in w.index:
            proxy[t] = bool(win[t].isna().any())
            paths[t] = betas[t] * win[market] if proxy[t] else win[t]
        cum = pd.DataFrame(paths).add(1).cumprod()
        value = (cum * w).sum(axis=1)                      # buy-and-hold com os pesos iniciais
        rp = value.pct_change().fillna(value.iloc[0] - 1)
        for t in w.index:
            st = _path_stats(paths[t])
            rows.append({"cenario": nome, "inicio": ini, "fim": fim, "serie": t, "proxy": proxy[t],
                         **st, "contrib": float(w[t] * st["perda"])})
        st_p = _path_stats(rp)
        rows.append({"cenario": nome, "inicio": ini, "fim": fim, "serie": "Carteira", "proxy": any(proxy.values()),
                     **st_p, "contrib": st_p["perda"],
                     "pior_dia_em_cvar": st_p["pior_dia"] / cvar_now if cvar_now else np.nan})
        rows.append({"cenario": nome, "inicio": ini, "fim": fim, "serie": market, "proxy": False,
                     **_path_stats(win[market]), "contrib": np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- I/O
def main() -> Path:
    prices = pd.read_parquet(CLEAN / "prices.parquet")
    simple = prices.pct_change().iloc[1:]
    sec = list(SECTORS)
    w = portfolio_weights(sec)

    start = prices.index.max() - pd.DateOffset(years=CVAR_YEARS)
    tab, resumo = risk_contributions(simple.loc[simple.index > start, sec], w)
    betas = pd.read_csv(FEATURES / "sector_metrics.csv", index_col=0)["beta"]   # fonte: risk_metrics
    stress = stress_test(simple, w, betas, cvar_now=resumo["cvar"])
    shrink = shrinkage_table(simple[sec].dropna())

    FEATURES.mkdir(parents=True, exist_ok=True)
    rc = pd.concat([tab, pd.DataFrame({"peso": [1.0], "contrib_vol": [resumo["vol"]], "pct_vol": [1.0],
                                       "contrib_cvar": [resumo["cvar"]], "pct_cvar": [1.0]},
                                      index=["Carteira"])]).rename_axis("setor")
    rc.to_csv(FEATURES / "risk_contribution.csv", float_format="%.6f")
    stress.to_csv(FEATURES / "stress_tests.csv", index=False, float_format="%.6f")
    shrink.to_csv(FEATURES / "covariance_shrinkage.csv", index=False, float_format="%.6f")

    print(f"[carteira] {resumo['inicio']} → {resumo['fim']}: vol {resumo['vol']:.2%}, "
          f"CVaR {CVAR_LEVEL:.1%} 1 dia {resumo['cvar']:.2%} ({resumo['dias_cauda']} dias de cauda), "
          f"diversificação vol {resumo['racio_diversificacao_vol']:.2f}× · CVaR {resumo['racio_diversificacao_cvar']:.2f}×")
    print(tab[["peso", "vol_individual", "pct_vol", "cvar_individual", "pct_cvar"]].round(3).to_string())
    print(shrink.round(4).to_string(index=False))
    piv = stress.pivot(index="serie", columns="cenario", values="perda").reindex(sec + ["Carteira", MARKET])
    print(piv.round(3).to_string())
    return FEATURES / "risk_contribution.csv"


if __name__ == "__main__":
    main()
