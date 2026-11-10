"""Backtesting do VaR a 1 dia (Sprint 2).

Funções puras (sem Streamlit, sem I/O) — o `main()` no fim é o único ponto com I/O:
  hist_var_forecast    VaR histórico rolante: quantil dos últimos `window` retornos até t-1
  garch_var_forecast   VaR GJR-GARCH(1,1)-t, re-estimado no 1.º dia útil de cada mês
  exceptions           série de exceções: perda real > VaR previsto
  kupiec               teste POF (proportion of failures) de Kupiec (1995)
  christoffersen       teste de independência de Christoffersen (1998)
  conditional_coverage teste condicional combinado (LR_cc = LR_pof + LR_ind)
  traffic_light        semáforo de Basileia nos últimos 250 dias
  backtest_table       tabela de resultados por setor, método e nível
  run                  VaR previstos + exceções + tabela para todos os setores

Convenções (CLAUDE.md): retornos simples e VaR em perda positiva. O GARCH é estimado
em retornos logarítmicos e o quantil é convertido para retorno simples (r = e^x - 1).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.special import xlogy
from scipy.stats import binom, chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (BT_LEVELS, BT_METHODS, BT_REFIT_FREQ, BT_SIGNIF,  # noqa: E402
                             BT_TL_DAYS, BT_TL_GREEN, BT_TL_RED, BT_WINDOW, CLEAN, FEATURES,
                             SECTORS)


# ----------------------------------------------------------------------------- previsões de VaR
def hist_var_forecast(r: pd.Series, level: float, window: int) -> pd.Series:
    """VaR histórico do dia t = -quantil(1 - level) dos `window` retornos de t-window a t-1.

    O `shift(1)` garante que o dia t não entra na sua própria previsão.
    """
    r = r.dropna()
    q = r.rolling(window, min_periods=window).quantile(1 - level)
    return (-q.shift(1)).rename(f"var_historico_{level}")


def refit_positions(index: pd.DatetimeIndex, window: int, freq: str) -> list[int]:
    """Posições onde o GARCH é re-estimado: a 1.ª com `window` observações anteriores e depois
    o 1.º dia útil de cada período `freq` (mensal por omissão)."""
    per = index.to_period(freq)
    return [i for i in range(window, len(index)) if i == window or per[i] != per[i - 1]]


def garch_var_forecast(r: pd.Series, levels: tuple[float, ...], window: int,
                       refit_freq: str = BT_REFIT_FREQ) -> pd.DataFrame:
    """VaR GJR-GARCH(1,1) com inovações t-Student, em perda positiva (retornos simples).

    Em cada data de re-estimação i, os parâmetros são estimados com as observações [i-window, i)
    (só passado). Até à re-estimação seguinte, os parâmetros ficam fixos e a variância condicional
    é atualizada dia a dia pela recursão do GARCH: a previsão para o dia t usa dados até t-1.
    """
    r = r.dropna()
    x = np.log1p(r) * 100                         # retornos log em %, como em risk_metrics.garch
    am = arch_model(x, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
    pos = refit_positions(x.index, window, refit_freq)
    out = np.full((len(x), len(levels)), np.nan)
    start_vals = None
    for k, i in enumerate(pos):
        end = pos[k + 1] if k + 1 < len(pos) else len(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(first_obs=i - window, last_obs=i, disp="off", starting_values=start_vals)
        start_vals = res.params.values
        # A linha s da previsão é a previsão para s+1 com informação até s.
        f = res.forecast(horizon=1, start=i - 1, reindex=False)
        mu = f.mean.iloc[: end - i, 0].to_numpy()
        sig = np.sqrt(f.variance.iloc[: end - i, 0].to_numpy())
        nu = res.params["nu"]
        for j, lv in enumerate(levels):
            q = am.distribution.ppf(1 - lv, np.array([nu]))  # quantil da t padronizada (variância 1)
            out[i:end, j] = -np.expm1((mu + sig * q) / 100)  # quantil log → perda simples positiva
    return pd.DataFrame(out, index=r.index, columns=[f"var_garch_{lv}" for lv in levels])


# ----------------------------------------------------------------------------- exceções
def exceptions(r: pd.Series, var: pd.Series) -> pd.Series:
    """1 quando a perda real (-r) excede o VaR previsto; só nos dias com previsão."""
    both = pd.concat([r, var], axis=1, join="inner").dropna()
    return (-both.iloc[:, 0] > both.iloc[:, 1]).astype(int).rename("excecao")


# ----------------------------------------------------------------------------- testes estatísticos
def kupiec(n_exc: int, n_obs: int, p: float) -> dict:
    """Teste POF de Kupiec: H0 taxa de exceções = p.

    LR_pof = -2 ln[(1-p)^(T-x) p^x] + 2 ln[(1-x/T)^(T-x) (x/T)^x]  ~  χ²(1)
    """
    x, t = n_exc, n_obs
    phat = x / t
    ll0 = xlogy(t - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(t - x, 1 - phat) + xlogy(x, phat)   # xlogy trata 0·ln 0 = 0 (x = 0 ou x = T)
    lr = float(-2 * (ll0 - ll1))
    return {"lr": lr, "pvalue": float(chi2.sf(lr, 1))}


def transition_counts(hits: pd.Series | np.ndarray) -> tuple[int, int, int, int]:
    """Contagens n_ij = nº de dias com estado i seguido de estado j (0 = sem exceção, 1 = exceção)."""
    h = np.asarray(hits, dtype=int)
    a, b = h[:-1], h[1:]
    return (int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum()),
            int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum()))


def christoffersen(hits: pd.Series | np.ndarray) -> dict:
    """Teste de independência de Christoffersen: H0 a probabilidade de exceção amanhã não depende
    de haver exceção hoje (π01 = π11).

    LR_ind = -2 ln[(1-π)^(n00+n10) π^(n01+n11)] + 2 ln[(1-π0)^n00 π0^n01 (1-π1)^n10 π1^n11]  ~  χ²(1)
    """
    n00, n01, n10, n11 = transition_counts(hits)
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi0) + xlogy(n01, pi0) + xlogy(n10, 1 - pi1) + xlogy(n11, pi1)
    lr = float(max(-2 * (ll0 - ll1), 0.0))
    return {"lr": lr, "pvalue": float(chi2.sf(lr, 1)), "n00": n00, "n01": n01, "n10": n10, "n11": n11}


def conditional_coverage(hits: pd.Series | np.ndarray, p: float) -> dict:
    """Teste condicional de Christoffersen: LR_cc = LR_pof + LR_ind  ~  χ²(2)."""
    h = np.asarray(hits, dtype=int)
    lr = kupiec(int(h.sum()), len(h), p)["lr"] + christoffersen(h)["lr"]
    return {"lr": lr, "pvalue": float(chi2.sf(lr, 2))}


def traffic_light_zones(level: float, days: int = BT_TL_DAYS, green: float = BT_TL_GREEN,
                        red: float = BT_TL_RED) -> tuple[int, int]:
    """Limites (máx. verde, máx. amarelo) em nº de exceções, pela binomial acumulada.

    A 99% e 250 dias devolve (4, 9): as zonas do Basel Committee (1996).
    """
    cdf = binom.cdf(np.arange(days + 1), days, 1 - level)
    max_green = int(np.argmax(cdf >= green)) - 1
    max_yellow = int(np.argmax(cdf >= red)) - 1
    return max_green, max_yellow


def traffic_light(hits: pd.Series, level: float, days: int = BT_TL_DAYS) -> tuple[str, int]:
    """Semáforo de Basileia com as exceções dos últimos `days` dias."""
    n = int(np.asarray(hits)[-days:].sum())
    g, y = traffic_light_zones(level, days)
    return ("verde" if n <= g else "amarelo" if n <= y else "vermelho"), n


# ----------------------------------------------------------------------------- tabela
def backtest_row(r: pd.Series, var: pd.Series, level: float) -> dict:
    hits = exceptions(r, var)
    n, x, p = len(hits), int(hits.sum()), 1 - level
    kp, ch, cc = kupiec(x, n, p), christoffersen(hits), conditional_coverage(hits, p)
    zona, x250 = traffic_light(hits, level)
    return {
        "inicio": str(hits.index[0].date()), "fim": str(hits.index[-1].date()), "dias": n,
        "excecoes": x, "esperadas": round(n * p, 1), "taxa_obs": x / n, "taxa_esp": p,
        "lr_kupiec": kp["lr"], "p_kupiec": kp["pvalue"],
        "lr_christoffersen": ch["lr"], "p_christoffersen": ch["pvalue"],
        "lr_cc": cc["lr"], "p_cc": cc["pvalue"],
        f"excecoes_{BT_TL_DAYS}": x250, "semaforo": zona,
    }


def sector_forecasts(r: pd.Series, levels: tuple[float, ...] = BT_LEVELS, window: int = BT_WINDOW,
                     refit_freq: str = BT_REFIT_FREQ) -> pd.DataFrame:
    """Retorno real + VaR previsto pelos dois métodos, só nos dias em que ambos existem."""
    r = r.dropna()
    cols = [hist_var_forecast(r, lv, window) for lv in levels]
    cols.append(garch_var_forecast(r, levels, window, refit_freq))
    return pd.concat([r.rename("ret"), *cols], axis=1).dropna()


def backtest_table(fc: pd.DataFrame, levels: tuple[float, ...] = BT_LEVELS,
                   methods: tuple[str, ...] = BT_METHODS) -> pd.DataFrame:
    """Uma linha por método e nível a partir da saída de `sector_forecasts`."""
    rows = [{"metodo": m, "nivel": lv} | backtest_row(fc["ret"], fc[f"var_{m}_{lv}"], lv)
            for m in methods for lv in levels]
    return pd.DataFrame(rows)


def run(simple: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devolve (tabela de resultados, séries diárias em formato longo) para os `tickers`."""
    tables, series = [], []
    for t in tickers:
        fc = sector_forecasts(simple[t])
        tables.append(backtest_table(fc).assign(setor=t))
        series.append(fc.assign(setor=t))
    res = pd.concat(tables, ignore_index=True)
    res = res[["setor", *[c for c in res.columns if c != "setor"]]]
    ser = pd.concat(series).rename_axis("data").reset_index()
    return res, ser


# ----------------------------------------------------------------------------- I/O
def main() -> Path:
    prices = pd.read_parquet(CLEAN / "prices.parquet")
    simple = prices.pct_change()
    t0 = time.time()
    res, ser = run(simple, list(SECTORS))
    FEATURES.mkdir(parents=True, exist_ok=True)
    out = FEATURES / "var_backtest.csv"
    res.to_csv(out, index=False, float_format="%.6f")
    ser.to_parquet(FEATURES / "var_backtest_series.parquet", index=False)
    print(f"[backtest] {len(res)} linhas em {time.time() - t0:.0f}s → {out.name} "
          f"(janela {BT_WINDOW} dias, GARCH re-estimado por período '{BT_REFIT_FREQ}', α = {BT_SIGNIF})")
    cols = ["setor", "metodo", "nivel", "dias", "excecoes", "taxa_obs", "taxa_esp",
            "p_kupiec", "p_christoffersen", "p_cc", f"excecoes_{BT_TL_DAYS}", "semaforo"]
    print(res[cols].round(4).to_string(index=False))
    return out


if __name__ == "__main__":
    main()
