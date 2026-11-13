"""VaR e ES a 10 e 21 dias úteis, por simulação de trajetórias do GARCH (Fase E; QRM 2.3.4, McNeil e Frey 2000).

A regra da raiz do tempo (VaR_h = √h · VaR_1) assume retornos iid. Com GARCH a volatilidade de hoje regressa à
média ao longo do horizonte, por isso simula-se dia a dia:
    r_d = μ + σ_d z_d,   σ²_{d+1} = ω + (α + γ·1[ε_d < 0]) ε_d² + β σ_d²,   ε_d = σ_d z_d
com z sorteado como no modelo oficial (EVT: reamostragem dos resíduos no corpo, Pareto na cauda). A perda a h dias
é −(exp(Σ r_d) − 1), em retorno simples e número positivo.

Funções puras: simulate_horizon, risk_measures, forecast, backtest_horizon. I/O em main().
Saídas: data/features/horizon_risk.csv (previsão de hoje) e horizon_backtest.csv (blocos sem sobreposição).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (BT_LEVELS, BT_SEED, BT_WINDOW, CLEAN, CVAR_LEVEL, FEATURES, HORIZON_BT_NSIM,  # noqa: E402
                             HORIZON_NSIM, HORIZONS, OFFICIAL_VAR_METHOD, SECTORS)
from src.collect.sessao import ultima_sessao_da_semana  # noqa: E402
from src.models import backtest as bt  # noqa: E402

RAIZ, SIMULACAO = "raiz", "simulacao"


def simulate_horizon(g: pd.Series, pool: np.ndarray, h: int, n_sim: int, rng: np.random.Generator,
                     method: str = OFFICIAL_VAR_METHOD) -> np.ndarray:
    """`n_sim` perdas simples a `h` dias. `g`: linha de `backtest.garch_filter` (mu, sigma do 1.º dia, ω, α, γ, β)."""
    z = bt.evt_draws(pool, (h, n_sim), rng) if method == "evt" else rng.choice(pool, size=(h, n_sim))
    s2 = np.full(n_sim, g["sigma"] ** 2)
    total = np.zeros(n_sim)
    for d in range(h):
        e = np.sqrt(s2) * z[d]
        total += g["mu"] + e
        s2 = g["omega"] + (g["alpha"] + g["gamma"] * (e < 0)) * e ** 2 + g["beta"] * s2
    return -np.expm1(total / 100)


def risk_measures(losses: np.ndarray, es_level: float = CVAR_LEVEL) -> dict:
    """var99, var975 e es975 de uma amostra de perdas (os nomes de `official_risk.csv`)."""
    v99, v975 = np.quantile(losses, [0.99, es_level])
    return {"var99": float(v99), "var975": float(v975), "es975": float(losses[losses >= v975].mean())}


def forecast(r: pd.Series, horizons: tuple[int, ...] = HORIZONS, n_sim: int = HORIZON_NSIM,
             seed: int = BT_SEED) -> pd.DataFrame:
    """Uma linha por horizonte, a partir da próxima sessão. `*_raiz` = medida a 1 dia × √h, para comparar."""
    g, pools = bt.garch_filter(bt.extend_next_day(r.dropna().tail(BT_WINDOW + 45)), BT_WINDOW)
    hoje = g.iloc[-1]
    pool = pools[int(hoje["refit"])]
    um = risk_measures(simulate_horizon(hoje, pool, 1, n_sim, np.random.default_rng(seed)))
    rows = []
    for h in horizons:
        m = risk_measures(simulate_horizon(hoje, pool, h, n_sim, np.random.default_rng(seed + h)))
        rows.append({"horizonte": h} | m | {f"{k}_raiz": v * np.sqrt(h) for k, v in um.items()})
    return pd.DataFrame(rows)


def backtest_horizon(r: pd.Series, h: int, n_sim: int = HORIZON_BT_NSIM, seed: int = BT_SEED,
                     levels: tuple[float, ...] = BT_LEVELS) -> pd.DataFrame:
    """Blocos de `h` dias sem sobreposição: o VaR previsto na véspera do bloco contra a perda do bloco.

    Compara a simulação com a regra da raiz do tempo (VaR a 1 dia do modelo oficial × √h). Uma linha por
    método × nível: blocos, exceções, exceções esperadas e p-valor de Kupiec."""
    r = r.dropna()
    g, pools = bt.garch_filter(r, BT_WINDOW)
    um_dia = bt.garch_risk(g, pools, levels)
    rng = np.random.default_rng(seed)
    hits = {(m, lv): [] for m in (SIMULACAO, RAIZ) for lv in levels}
    for t0 in range(int(np.flatnonzero(g["sigma"].notna())[0]), len(r) - h + 1, h):
        perda = -(np.prod(1 + r.iloc[t0:t0 + h].to_numpy()) - 1)
        sim = simulate_horizon(g.iloc[t0], pools[int(g["refit"].iloc[t0])], h, n_sim, rng)
        for lv in levels:
            hits[(SIMULACAO, lv)].append(perda > np.quantile(sim, lv))
            hits[(RAIZ, lv)].append(perda > um_dia[f"var_{OFFICIAL_VAR_METHOD}_{lv}"].iloc[t0] * np.sqrt(h))
    rows = []
    for (m, lv), x in hits.items():
        n, exc = len(x), int(np.sum(x))
        rows.append({"horizonte": h, "metodo": m, "nivel": lv, "blocos": n, "excecoes": exc,
                     "esperadas": n * (1 - lv), "p_kupiec": bt.kupiec(exc, n, 1 - lv)["pvalue"]})
    return pd.DataFrame(rows)


def main(full: bool | None = None) -> Path:
    simple = pd.read_parquet(CLEAN / "prices.parquet")[list(SECTORS)].pct_change()
    FEATURES.mkdir(parents=True, exist_ok=True)
    out, bt_f = FEATURES / "horizon_risk.csv", FEATURES / "horizon_backtest.csv"
    prev = pd.concat([forecast(simple[t]).assign(setor=t) for t in SECTORS], ignore_index=True)
    prev.assign(dados_ate=str(simple.index.max().date())).to_csv(out, index=False, float_format="%.6f")
    if full is None:                                     # backtest: 1.ª vez e na última sessão da semana
        full = not bt_f.exists() or ultima_sessao_da_semana(simple.index.max().date())
    if full:
        t0 = time.time()
        res = pd.concat([backtest_horizon(simple[t], h, seed=BT_SEED + k).assign(setor=t)
                         for k, t in enumerate(SECTORS) for h in HORIZONS], ignore_index=True)
        res.to_csv(bt_f, index=False, float_format="%.6f")
        tot = res.groupby(["horizonte", "metodo", "nivel"])[["blocos", "excecoes", "esperadas"]].sum()
        print(f"[horizonte] backtest em {time.time() - t0:.0f}s (blocos sem sobreposição, 11 setores somados):")
        print(tot.round(1).to_string())
    med = prev.groupby("horizonte")[["es975", "es975_raiz"]].median()
    print("[horizonte] ES 97,5% mediano dos setores: "
          + "; ".join(f"{h} dias {v.es975:.1%} (raiz do tempo {v.es975_raiz:.1%})" for h, v in med.iterrows()))
    return out


if __name__ == "__main__":
    main(full=True if "--completo" in sys.argv else None)
