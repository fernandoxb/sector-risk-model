"""Backtesting do VaR e do Expected Shortfall a 1 dia (Sprint 2).

Funções puras (sem Streamlit, sem I/O) — o `main()` no fim é o único ponto com I/O:
  hist_forecast        VaR e ES históricos rolantes: últimos `window` retornos até t-1
  garch_filter         GJR-GARCH(1,1)-t re-estimado no 1.º dia útil de cada mês → μ, σ, ν por dia
  garch_risk           VaR e ES a partir do filtro: t-Student ("garch") e FHS ("fhs")
  exceptions           série de exceções: perda real > VaR previsto
  kupiec               teste POF (proportion of failures) de Kupiec (1995)
  christoffersen       teste de independência de Christoffersen (1998)
  conditional_coverage teste condicional combinado (LR_cc = LR_pof + LR_ind)
  traffic_light        semáforo de Basileia nos últimos 250 dias
  frtb_desk            backtest por mesa do FRTB (máx. de exceções a 99% e 97,5%)
  z2 / z2_pvalue       teste Z2 do ES de Acerbi–Székely (2014), p-valor por simulação sob H0
  simulate_losses      perdas simuladas sob H0 (a distribuição prevista de cada dia)
  backtest_table       tabela de resultados por setor, método e nível
  next_day_forecast    VaR/ES do método oficial para o próximo dia útil (mesmo código do backtest)
  run                  previsões + tabela + previsão oficial de amanhã para todos os setores

Convenções (CLAUDE.md): retornos simples, VaR e ES em perda positiva. O GARCH é estimado
em retornos logarítmicos (em %); cada cenário x passa a retorno simples por r = e^x - 1.
O ES histórico usa `src/models/cockpit.py::cvar` (fonte única do CVaR).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import StudentsT
from scipy.special import xlogy
from functools import lru_cache

from scipy.stats import binom, chi2, genpareto

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (BT_ES_GRID, BT_ES_NSIM, BT_FRTB_MAX, BT_LEVELS,  # noqa: E402
                             BT_METHODS, BT_REFIT_FREQ, BT_SEED, BT_SIGNIF, BT_TL_DAYS,
                             BT_MF_NBOOT, BT_SPLIT, BT_SPLIT_MIN_DAYS, BT_TL_GREEN, BT_TL_RED, BT_WINDOW, CLEAN,
                             EVT_TAIL_FRAC, EVT_XI_MAX,
                             CVAR_LEVEL, FEATURES, OFFICIAL_MEASURES, OFFICIAL_VAR_METHOD, SECTORS)
from src.collect.sessao import proximo_dia_util  # noqa: E402
from src.models.cockpit import cvar  # noqa: E402

T_DIST = StudentsT()   # t padronizada (variância 1), a mesma do arch_model(dist="t")


# ----------------------------------------------------------------------------- histórico
def hist_forecast(r: pd.Series, levels: tuple[float, ...], window: int) -> pd.DataFrame:
    """VaR e ES históricos do dia t com os `window` retornos de t-window a t-1.

    VaR = -quantil(1 - nível); ES = cockpit.cvar na mesma janela. O `shift(1)` garante
    que o dia t não entra na sua própria previsão.
    """
    r = r.dropna()
    roll = r.rolling(window, min_periods=window)
    cols = {}
    for lv in levels:
        cols[f"var_historico_{lv}"] = -roll.quantile(1 - lv).shift(1)
        cols[f"es_historico_{lv}"] = roll.apply(lambda w, a=lv: cvar(w, a), raw=False).shift(1)
    return pd.DataFrame(cols, index=r.index)


# ----------------------------------------------------------------------------- GARCH
def refit_positions(index: pd.DatetimeIndex, window: int, freq: str) -> list[int]:
    """Posições onde o GARCH é re-estimado: a 1.ª com `window` observações anteriores e depois
    o 1.º dia útil de cada período `freq` (mensal por omissão)."""
    per = index.to_period(freq)
    return [i for i in range(window, len(index)) if i == window or per[i] != per[i - 1]]


GARCH_PARAMS = ["omega", "alpha[1]", "gamma[1]", "beta[1]"]


def garch_filter(r: pd.Series, window: int,
                 refit_freq: str = BT_REFIT_FREQ) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """GJR-GARCH(1,1) com inovações t-Student em janela móvel, sem look-ahead.

    Em cada data de re-estimação i, os parâmetros são estimados com [i-window, i). Até à
    re-estimação seguinte ficam fixos e σ é atualizada dia a dia pela recursão: a previsão
    para o dia t usa dados até t-1.

    Devolve (por dia: mu, sigma em % log, nu, refit = posição da estimação usada, e os parâmetros ω, α, γ, β) e
    {refit: resíduos padronizados da janela de estimação} para a FHS.
    """
    r = r.dropna()
    x = np.log1p(r) * 100
    am = arch_model(x, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
    pos = refit_positions(x.index, window, refit_freq)
    out = np.full((len(x), 8), np.nan)
    pools: dict[int, np.ndarray] = {}
    start_vals = None
    for k, i in enumerate(pos):
        end = pos[k + 1] if k + 1 < len(pos) else len(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(first_obs=i - window, last_obs=i, disp="off", starting_values=start_vals)
        start_vals = res.params.values
        f = res.forecast(horizon=1, start=i - 1, reindex=False)   # linha s = previsão para s+1
        out[i:end, 0] = f.mean.iloc[: end - i, 0].to_numpy()
        out[i:end, 1] = np.sqrt(f.variance.iloc[: end - i, 0].to_numpy())
        out[i:end, 2] = res.params["nu"]
        out[i:end, 3] = i
        out[i:end, 4:] = res.params[GARCH_PARAMS].to_numpy()       # para simular trajetórias (horizonte.py)
        pools[i] = res.std_resid.dropna().to_numpy()
    g = pd.DataFrame(out, index=r.index, columns=["mu", "sigma", "nu", "refit", "omega", "alpha", "gamma", "beta"])
    return g, pools


def t_tail_grid(level: float, nu: float, n: int = BT_ES_GRID) -> np.ndarray:
    """Quantis da t padronizada nos pontos médios de n fatias iguais da cauda (0, 1 - nível).

    A média de uma função sobre estes pontos aproxima E[f(z) | z <= q] (regra do ponto médio).
    """
    p = 1 - level
    return T_DIST.ppf(p * (np.arange(n) + 0.5) / n, np.array([nu]))


# ----------------------------------------------------------------------------- EVT: cauda de Pareto dos resíduos
@lru_cache(maxsize=8192)
def _gpd_fit(buf: bytes, frac: float, xi_max: float) -> tuple[float, float, float, float]:
    y = np.sort(-np.frombuffer(buf, dtype=float))            # perdas padronizadas, por ordem crescente
    k = max(int(np.floor(frac * len(y))), 10)
    u = float(y[-k - 1])                                     # limiar: deixa k excessos acima
    exc = y[-k:] - u
    try:
        xi, _, beta = genpareto.fit(exc, floc=0)
        if not (np.isfinite(xi) and np.isfinite(beta) and beta > 0):
            raise ValueError
    except Exception:  # noqa: BLE001  ajuste falhou: cauda exponencial (ξ = 0) com a média dos excessos
        xi, beta = 0.0, float(exc.mean())
    return u, float(np.clip(xi, -0.5, xi_max)), float(beta), k / len(y)


def gpd_tail(z: np.ndarray, frac: float = EVT_TAIL_FRAC, xi_max: float = EVT_XI_MAX) -> tuple[float, float, float, float]:
    """Cauda das perdas padronizadas (−z) por máxima verosimilhança: (limiar u, ξ, β, fração na cauda).

    Os excessos acima de u seguem uma Pareto generalizada G_{ξ,β} (QRM, teorema 7.20). Guardado em cache por
    conjunto de resíduos, porque o mesmo ajuste serve a previsão e a simulação."""
    return _gpd_fit(np.ascontiguousarray(z, dtype=float).tobytes(), frac, xi_max)


def gpd_quantile(prob_cauda, u: float, xi: float, beta: float, frac: float) -> np.ndarray:
    """Perda padronizada ultrapassada com probabilidade `prob_cauda` (<= frac), QRM eq. 7.18:
    q = u + β/ξ · ((prob/frac)^(−ξ) − 1); no limite ξ → 0, u − β ln(prob/frac)."""
    ratio = np.asarray(prob_cauda, dtype=float) / frac
    if abs(xi) < 1e-9:
        return u - beta * np.log(ratio)
    return u + beta / xi * (ratio ** (-xi) - 1)


def gpd_es(prob_cauda: float, u: float, xi: float, beta: float, frac: float) -> float:
    """Expected shortfall da perda padronizada, QRM eq. 7.19: ES = q/(1 − ξ) + (β − ξu)/(1 − ξ), para ξ < 1."""
    return float(gpd_quantile(prob_cauda, u, xi, beta, frac) / (1 - xi) + (beta - xi * u) / (1 - xi))


def evt_draws(pool: np.ndarray, size, rng: np.random.Generator) -> np.ndarray:
    """Resíduos sorteados: reamostragem no corpo e, na cauda (fração EVT_TAIL_FRAC), da Pareto ajustada."""
    u, xi, beta, frac = gpd_tail(pool)
    z = rng.choice(pool[-pool <= u], size=size)
    na_cauda = rng.random(size) < frac
    z[na_cauda] = -gpd_quantile(frac * rng.random(int(na_cauda.sum())), u, xi, beta, frac)
    return z


def _to_loss(mu: np.ndarray, sig: np.ndarray, z: np.ndarray | float) -> np.ndarray:
    """Cenário log (%) μ + σz → perda simples positiva."""
    return -np.expm1((mu + sig * z) / 100)


def garch_risk(g: pd.DataFrame, pools: dict[int, np.ndarray],
               levels: tuple[float, ...], n_grid: int = BT_ES_GRID) -> pd.DataFrame:
    """VaR e ES em perda positiva a partir de μ, σ previstos:

    garch  quantil e cauda da t padronizada com o ν estimado
    fhs    quantil e cauda empíricos dos resíduos padronizados da janela de estimação
    evt    quantil e cauda de uma Pareto generalizada ajustada aos piores resíduos (EVT condicional)
    """
    cols = [f"{m}_{k}_{lv}" for k in ("garch", "fhs", "evt") for lv in levels for m in ("var", "es")]
    out = pd.DataFrame(np.nan, index=g.index, columns=cols)
    for i, blk in g.dropna().groupby("refit"):
        mu, sig = blk["mu"].to_numpy()[:, None], blk["sigma"].to_numpy()[:, None]
        nu, z = blk["nu"].iloc[0], pools[int(i)]
        for lv in levels:
            q_t = T_DIST.ppf(1 - lv, np.array([nu]))
            q_e = np.quantile(z, 1 - lv)                 # interpolação linear, como o histórico
            out.loc[blk.index, f"var_garch_{lv}"] = _to_loss(mu, sig, q_t)[:, 0]
            out.loc[blk.index, f"es_garch_{lv}"] = _to_loss(mu, sig, t_tail_grid(lv, nu, n_grid)).mean(1)
            out.loc[blk.index, f"var_fhs_{lv}"] = _to_loss(mu, sig, q_e)[:, 0]
            out.loc[blk.index, f"es_fhs_{lv}"] = _to_loss(mu, sig, z[z <= q_e]).mean(1)
            gp = gpd_tail(z)
            p = 1 - lv
            out.loc[blk.index, f"var_evt_{lv}"] = _to_loss(mu, sig, -gpd_quantile(p, *gp))[:, 0]
            # shortcut: a média por fatias fica curta com ξ alto (cerca de 5% a ξ = 0,6; nos setores ξ mediano é 0,13);
            # se aparecerem ξ > 0,6 com frequência, integrar a última fatia com a fórmula fechada.
            grelha = -gpd_quantile(p * (np.arange(n_grid) + 0.5) / n_grid, *gp)
            out.loc[blk.index, f"es_evt_{lv}"] = _to_loss(mu, sig, grelha).mean(1)
    return out


# ----------------------------------------------------------------------------- exceções
def exceptions(r: pd.Series, var: pd.Series) -> pd.Series:
    """1 quando a perda real (-r) excede o VaR previsto; só nos dias com previsão."""
    both = pd.concat([r, var], axis=1, join="inner").dropna()
    return (-both.iloc[:, 0] > both.iloc[:, 1]).astype(int).rename("excecao")


# ----------------------------------------------------------------------------- testes do VaR
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


def frtb_desk(n_exc_250: int, level: float) -> str:
    """Backtest por mesa do FRTB: reprovada com mais de 12 exceções a 99% ou 30 a 97,5% em 250 dias."""
    return "aprovada" if n_exc_250 <= BT_FRTB_MAX[level] else "reprovada"


# ----------------------------------------------------------------------------- teste do ES
def z2(loss: np.ndarray, var: np.ndarray, es: np.ndarray, p: float) -> np.ndarray:
    """Estatística Z2 de Acerbi–Székely (2014), em perda positiva:

        Z2 = 1 - Σ_t L_t·1{L_t > VaR_t} / (T · p · ES_t)

    Sob H0 (ES bem previsto) E[Z2] = 0; Z2 < 0 indica ES subestimado.
    `loss` pode ser (T,) ou (n_sim, T); devolve um valor por linha.
    """
    loss = np.asarray(loss, dtype=float)
    hit = loss > var
    return 1 - (np.where(hit, loss, 0.0) / es).sum(axis=-1) / (loss.shape[-1] * p)


def mcneil_frey(loss: np.ndarray, var: np.ndarray, es: np.ndarray, n_boot: int = BT_MF_NBOOT,
                seed: int = BT_SEED) -> dict:
    """Teste do ES pelos resíduos das exceções (McNeil e Frey 2000; QRM 4.4.3, eq. 4.52).

    Nos dias de exceção, R_t = (L_t − ES_t) / ES_t deve ter média zero se o ES estiver certo. H1: média > 0
    (ES subestimado). P-valor por bootstrap da estatística t com os resíduos centrados (Efron e Tibshirani).
    Ao contrário do Z2, olha só para o tamanho das perdas nas exceções, não para a sua frequência.
    Devolve {"n", "media", "pvalue"}; sem pelo menos 5 exceções, pvalue = NaN.
    """
    loss, var, es = (np.asarray(a, dtype=float) for a in (loss, var, es))
    hit = loss > var
    res = (loss[hit] - es[hit]) / es[hit]
    n = len(res)
    if n < 5 or res.std(ddof=1) == 0:
        return {"n": n, "media": float(res.mean()) if n else float("nan"), "pvalue": float("nan")}
    t_obs = res.mean() / (res.std(ddof=1) / np.sqrt(n))
    rng = np.random.default_rng(seed)
    amostra = rng.choice(res - res.mean(), size=(n_boot, n))
    with np.errstate(divide="ignore", invalid="ignore"):         # reamostra sem variação: t fica inf/NaN e não conta
        t_boot = amostra.mean(1) / (amostra.std(1, ddof=1) / np.sqrt(n))
    return {"n": n, "media": float(res.mean()), "pvalue": float((np.sum(t_boot >= t_obs) + 1) / (n_boot + 1))}


def z2_pvalue(z_obs: float, z_sim: np.ndarray) -> float:
    """P-valor unilateral (ES subestimado): fração das simulações sob H0 com Z2 <= observado."""
    return float((np.sum(z_sim <= z_obs) + 1) / (len(z_sim) + 1))


def simulate_losses(method: str, r: pd.Series, aux: pd.DataFrame, pools: dict[int, np.ndarray],
                    window: int, n_sim: int, rng: np.random.Generator) -> np.ndarray:
    """Perdas (n_sim, T) simuladas sob H0: cada dia tirado da distribuição que o método previu.

    historico  reamostragem uniforme dos `window` retornos anteriores
    garch      μ + σ·z, z ~ t padronizada com o ν do dia
    fhs        μ + σ·z, z reamostrado dos resíduos padronizados da estimação em vigor
    evt        como a FHS no corpo da distribuição; na cauda (fração EVT_TAIL_FRAC), z tirado da Pareto ajustada
    """
    T = len(aux)
    if method == "historico":
        pos = aux["pos"].to_numpy().astype(int)
        idx = pos - 1 - rng.integers(0, window, size=(n_sim, T))
        return -r.dropna().to_numpy()[idx]
    mu, sig = aux["mu"].to_numpy(), aux["sigma"].to_numpy()
    if method == "garch":
        nu = aux["nu"].to_numpy()
        z = rng.standard_t(nu, size=(n_sim, T)) * np.sqrt((nu - 2) / nu)
    elif method == "fhs":
        z = np.empty((n_sim, T))
        refit = aux["refit"].to_numpy().astype(int)
        for i in np.unique(refit):
            cols = np.flatnonzero(refit == i)
            z[:, cols] = rng.choice(pools[i], size=(n_sim, len(cols)))
    elif method == "evt":
        z = np.empty((n_sim, T))
        refit = aux["refit"].to_numpy().astype(int)
        for i in np.unique(refit):
            cols = np.flatnonzero(refit == i)
            z[:, cols] = evt_draws(pools[i], (n_sim, len(cols)), rng)
    else:
        raise ValueError(f"Método desconhecido: {method}")
    return _to_loss(mu, sig, z)


# ----------------------------------------------------------------------------- tabela
def sector_forecasts(r: pd.Series, levels: tuple[float, ...] = BT_LEVELS, window: int = BT_WINDOW,
                     refit_freq: str = BT_REFIT_FREQ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(previsões, auxiliares, pools), só nos dias em que todos os métodos têm previsão.

    previsões: ret + var_/es_<método>_<nível>; auxiliares: pos, mu, sigma, nu, refit (para simular).
    """
    r = r.dropna()
    g, pools = garch_filter(r, window, refit_freq)
    fc = pd.concat([r.rename("ret"), hist_forecast(r, levels, window), garch_risk(g, pools, levels)],
                   axis=1).dropna()
    aux = g.assign(pos=np.arange(len(r))).loc[fc.index]
    return fc, aux, pools


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
        f"excecoes_{BT_TL_DAYS}": x250, "semaforo": zona, "frtb_mesa": frtb_desk(x250, level),
    }


def backtest_table(fc: pd.DataFrame, aux: pd.DataFrame, pools: dict, r: pd.Series,
                   levels: tuple[float, ...] = BT_LEVELS, methods: tuple[str, ...] = BT_METHODS,
                   window: int = BT_WINDOW, n_sim: int = BT_ES_NSIM, seed: int = BT_SEED) -> pd.DataFrame:
    """Uma linha por método e nível: testes do VaR + teste Z2 do ES."""
    rng = np.random.default_rng(seed)
    loss = -fc["ret"].to_numpy()
    rows = []
    for m in methods:
        sims = simulate_losses(m, r, aux, pools, window, n_sim, rng)
        for lv in levels:
            var, es = fc[f"var_{m}_{lv}"].to_numpy(), fc[f"es_{m}_{lv}"].to_numpy()
            z_obs = float(z2(loss, var, es, 1 - lv))
            rows.append({"metodo": m, "nivel": lv}
                        | backtest_row(fc["ret"], fc[f"var_{m}_{lv}"], lv)
                        | {"z2": z_obs, "p_z2": z2_pvalue(z_obs, z2(sims, var, es, 1 - lv)),
                           "p_mf": mcneil_frey(loss, var, es, seed=seed)["pvalue"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- previsão oficial
def extend_next_day(r: pd.Series) -> pd.Series:
    """Acrescenta a próxima sessão da NYSE (dias úteis sem feriados, src/collect/sessao.py) com retorno 0.

    O valor é um marcador: a previsão para esse dia só usa dados até à véspera (sem look-ahead,
    ver tests/test_backtest.py), por isso nunca entra no cálculo.
    """
    r = r.dropna()
    nxt = pd.Timestamp(proximo_dia_util(r.index[-1].date()))
    return pd.concat([r, pd.Series([0.0], index=pd.DatetimeIndex([nxt]))]).rename(r.name)


def official_row(fc: pd.DataFrame, method: str = OFFICIAL_VAR_METHOD,
                 measures: dict[str, tuple[str, float]] = OFFICIAL_MEASURES) -> dict:
    """Medidas oficiais (ex.: var99, var975, es975) da última linha de `sector_forecasts`."""
    row = fc.iloc[-1]
    return {"data_previsao": str(fc.index[-1].date()), "metodo": method,
            **{k: float(row[f"{m}_{method}_{lv}"]) for k, (m, lv) in measures.items()}}


def next_day_forecast(r: pd.Series, method: str = OFFICIAL_VAR_METHOD, levels: tuple[float, ...] = BT_LEVELS,
                      window: int = BT_WINDOW, refit_freq: str = BT_REFIT_FREQ) -> dict:
    """VaR e ES do método oficial para o próximo dia útil, com todos os dados de `r`."""
    fc, _, _ = sector_forecasts(extend_next_day(r), levels, window, refit_freq)
    return {"dados_ate": str(r.dropna().index[-1].date())} | official_row(fc, method)


def run(simple: pd.DataFrame, tickers: list[str], seed: int = BT_SEED,
        split: str | None = None, min_days: int = BT_SPLIT_MIN_DAYS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devolve (tabela de resultados, séries diárias em formato longo, previsão oficial de amanhã).

    Cada setor corre `sector_forecasts` uma só vez sobre a série estendida: a última linha é a
    previsão de amanhã e as restantes são exatamente o backtest (o marcador não as afeta).

    Com `split` (data), a tabela ganha a coluna `periodo`: "total", "selecao" (previsões antes de `split`) e
    "confirmacao" (de `split` em diante). As previsões são as mesmas; só muda a amostra dos testes. Um setor
    com menos de `min_days` previsões num período não entra nesse período.
    """
    tables, series, official = [], [], []
    for k, t in enumerate(tickers):
        r = simple[t].dropna()
        fc_ext, aux, pools = sector_forecasts(extend_next_day(r))
        official.append({"setor": t, "dados_ate": str(r.index[-1].date())} | official_row(fc_ext))
        fc, aux = fc_ext.iloc[:-1], aux.iloc[:-1]
        tab = backtest_table(fc, aux, pools, r, seed=seed + k).assign(setor=t)
        if split is not None:
            tab = tab.assign(periodo="total")
            antes = fc.index < pd.Timestamp(split)
            for nome, m in (("selecao", antes), ("confirmacao", ~antes)):
                if m.sum() >= min_days:
                    tab = pd.concat([tab, backtest_table(fc[m], aux[m], pools, r, seed=seed + k)
                                     .assign(setor=t, periodo=nome)], ignore_index=True)
        tables.append(tab)
        series.append(fc.assign(setor=t))
    res = pd.concat(tables, ignore_index=True)
    res = res[["setor", *[c for c in res.columns if c != "setor"]]]
    ser = pd.concat(series).rename_axis("data").reset_index()
    return res, ser, pd.DataFrame(official)


# ----------------------------------------------------------------------------- escolha do método e testes múltiplos
TESTES_ESCOLHA = ("p_kupiec", "p_christoffersen", "p_z2")       # frequência, independência e ES


def passes(res: pd.DataFrame, signif: float = BT_SIGNIF) -> pd.DataFrame:
    """Por método e nível: nº de séries que passam cada teste, nº que passam os três e nº de séries."""
    ok = res.assign(**{c: res[c] >= signif for c in TESTES_ESCOLHA})
    ok["tudo"] = ok[list(TESTES_ESCOLHA)].all(axis=1)
    out = ok.groupby(["metodo", "nivel"])[[*TESTES_ESCOLHA, "tudo"]].sum()
    out["series"] = ok.groupby(["metodo", "nivel"]).size()
    return out.reset_index()


def choose_method(res: pd.DataFrame, level: float = CVAR_LEVEL, signif: float = BT_SIGNIF) -> str:
    """Método com mais séries a passar os três testes ao nível `level`; em empate, o que passa mais no outro nível."""
    p = passes(res, signif)
    principal = p[np.isclose(p["nivel"], level)].set_index("metodo")["tudo"]
    outro = p[~np.isclose(p["nivel"], level)].groupby("metodo")["tudo"].sum().reindex(principal.index).fillna(0)
    return max(principal.index, key=lambda m: (principal[m], outro[m]))


def holm(pvals, signif: float = BT_SIGNIF) -> np.ndarray:
    """Correção de Holm (1979) para testes múltiplos: devolve True onde H0 continua rejeitada.

    Ordena os p-valores; rejeita o i-ésimo menor (i = 0, 1, …) enquanto p_(i) <= signif / (m − i) e pára no
    primeiro que falha. Controla a probabilidade de pelo menos um falso positivo, sem assumir independência.
    """
    p = np.asarray(pvals, dtype=float)
    ordem = np.argsort(p)
    rej = np.zeros(len(p), dtype=bool)
    for i, j in enumerate(ordem):
        if p[j] > signif / (len(p) - i):
            break
        rej[j] = True
    return rej


def multiple_testing(res: pd.DataFrame, signif: float = BT_SIGNIF) -> pd.DataFrame:
    """Por método: nº de testes (séries × níveis × 4 testes), rejeições a `signif`, as esperadas por acaso se o
    modelo estivesse certo, e as que sobrevivem à correção de Holm."""
    cols = ["p_kupiec", "p_christoffersen", "p_cc", "p_z2"]
    rows = []
    for m, g in res.groupby("metodo"):
        p = g[cols].to_numpy().ravel()
        rows.append({"metodo": m, "testes": len(p), "rejeicoes": int((p < signif).sum()),
                     "esperadas_por_acaso": round(len(p) * signif, 1), "rejeicoes_holm": int(holm(p, signif).sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- I/O
def main() -> Path:
    prices = pd.read_parquet(CLEAN / "prices.parquet")
    simple = prices.pct_change()
    t0 = time.time()
    res_all, ser, official = run(simple, list(SECTORS), split=BT_SPLIT)
    res = res_all[res_all["periodo"] == "total"].drop(columns="periodo").reset_index(drop=True)
    FEATURES.mkdir(parents=True, exist_ok=True)
    out = FEATURES / "var_backtest.csv"
    res.to_csv(out, index=False, float_format="%.6f")
    res_all[res_all["periodo"] != "total"].to_csv(FEATURES / "var_backtest_split.csv", index=False, float_format="%.6f")
    multiple_testing(res).to_csv(FEATURES / "var_backtest_multiple.csv", index=False)
    ser.to_parquet(FEATURES / "var_backtest_series.parquet", index=False)
    official.to_csv(FEATURES / "official_risk.csv", index=False, float_format="%.6f")
    print(f"[backtest] {len(res)} linhas em {time.time() - t0:.0f}s → {out.name} "
          f"(janela {BT_WINDOW} dias, GARCH re-estimado por período '{BT_REFIT_FREQ}', α = {BT_SIGNIF})")
    cols = ["setor", "metodo", "nivel", "excecoes", "esperadas", "p_kupiec", "p_christoffersen",
            "p_cc", "z2", "p_z2", f"excecoes_{BT_TL_DAYS}", "semaforo", "frtb_mesa"]
    print(res[cols].round(3).to_string(index=False))
    sel, conf = (res_all[res_all["periodo"] == k] for k in ("selecao", "confirmacao"))
    escolhido = choose_method(sel)
    print(f"[backtest] escolha com as previsões antes de {BT_SPLIT}: {escolhido} "
          f"({'igual ao' if escolhido == OFFICIAL_VAR_METHOD else 'DIFERENTE do'} método oficial)")
    print(passes(sel).assign(periodo="selecao").to_string(index=False))
    print(passes(conf).assign(periodo="confirmacao").to_string(index=False))
    print(multiple_testing(res).to_string(index=False))
    print(f"[backtest] previsão oficial ({OFFICIAL_VAR_METHOD}) para {official['data_previsao'].iloc[0]} "
          f"→ official_risk.csv")
    print(official.round(4).to_string(index=False))
    return out


if __name__ == "__main__":
    main()
