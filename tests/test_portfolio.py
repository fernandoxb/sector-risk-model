"""Testes da visão conjunta (src/models/portfolio.py).

Correr: .venv/bin/python -m pytest
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from src.models import portfolio as pf
from src.models.cockpit import cvar

TICK = ["A", "B", "C", "D", "E"]


@pytest.fixture(scope="module")
def rets() -> pd.DataFrame:
    """Retornos correlacionados (1 fator + idiossincrático) com caudas t."""
    rng = np.random.default_rng(7)
    n = 900
    f = rng.standard_t(4, n) * 0.008
    b = np.array([0.6, 0.9, 1.1, 1.3, 0.8])
    e = rng.standard_t(5, (n, 5)) * np.array([0.004, 0.006, 0.007, 0.01, 0.005])
    return pd.DataFrame(f[:, None] * b + e, index=pd.bdate_range("2019-01-01", periods=n), columns=TICK)


# ----------------------------------------------------------------------------- Ledoit-Wolf
@pytest.mark.parametrize("t", [30, 63, 400])
def test_ledoit_wolf_igual_ao_sklearn(rets, t):
    x = rets.iloc[:t].to_numpy()
    cov, delta = pf.ledoit_wolf(x)
    ref = LedoitWolf().fit(x)
    assert delta == pytest.approx(ref.shrinkage_, rel=1e-10)
    np.testing.assert_allclose(cov, ref.covariance_, rtol=1e-10)


def test_ledoit_wolf_melhora_o_condicionamento(rets):
    x = rets.iloc[:30].to_numpy()
    assert np.linalg.cond(pf.ledoit_wolf(x)[0]) < np.linalg.cond(pf.sample_cov(x))


def test_variancia_minima_diagonal():
    """Com Σ diagonal, w_i ∝ 1/σ_i²."""
    v = np.array([1.0, 4.0, 2.0])
    inv = 1 / v
    np.testing.assert_allclose(pf.min_variance_weights(np.diag(v)), inv / inv.sum())


def test_variancia_minima_e_minima(rets):
    cov = pf.sample_cov(rets.to_numpy())
    w = pf.min_variance_weights(cov)
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = rng.normal(size=len(w))
        d -= d.mean()                                  # perturbação que mantém Σw = 1
        w2 = w + 0.05 * d
        assert w @ cov @ w <= w2 @ cov @ w2


def test_oos_sem_look_ahead(rets):
    """Mudar os retornos a partir do dia t não altera os retornos fora da amostra antes de t."""
    a = pf.oos_min_variance(rets, 100, "M")
    t = 400
    choque = rets.copy()
    choque.iloc[t:] *= -4
    b = pf.oos_min_variance(choque, 100, "M")
    dia = rets.index[t - 1]
    pd.testing.assert_frame_equal(a.loc[:dia], b.loc[:dia])


# ----------------------------------------------------------------------------- contribuições
def test_euler_soma_o_risco_da_carteira(rets):
    w = pd.Series([0.1, 0.3, 0.2, 0.15, 0.25], index=TICK)
    tab, res = pf.risk_contributions(rets, w, 0.975)
    assert tab["contrib_vol"].sum() == pytest.approx(res["vol"], rel=1e-12)
    assert tab["contrib_cvar"].sum() == pytest.approx(res["cvar"], rel=1e-12)
    assert res["cvar"] == pytest.approx(cvar(rets @ w, 0.975))       # mesma função do mapa e do cockpit
    assert tab["pct_vol"].sum() == pytest.approx(1) and tab["pct_cvar"].sum() == pytest.approx(1)


def test_contribuicao_vol_igual_a_derivada_numerica(rets):
    """RC_i = w_i ∂σ_p/∂w_i, comparado com diferenças finitas centradas."""
    w = pd.Series([0.1, 0.3, 0.2, 0.15, 0.25], index=TICK)
    tab, _ = pf.risk_contributions(rets, w, 0.975)
    cov = pf.ledoit_wolf(rets.to_numpy())[0]
    h = 1e-6
    for i, t in enumerate(TICK):
        up, dn = w.to_numpy().copy(), w.to_numpy().copy()
        up[i] += h
        dn[i] -= h
        grad = (np.sqrt(up @ cov @ up) - np.sqrt(dn @ cov @ dn)) / (2 * h)
        assert tab.loc[t, "contrib_vol"] == pytest.approx(w[t] * grad * pf.ANN, rel=1e-6)


def test_pesos():
    assert pf.portfolio_weights(TICK).sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        pf.portfolio_weights(TICK, {"A": 0.5, "B": 0.5})


# ----------------------------------------------------------------------------- stress tests
def test_stress_buy_and_hold_a_mao():
    idx = pd.bdate_range("2020-01-01", periods=4)
    simple = pd.DataFrame({"A": [0.0, -0.10, -0.20, 0.05],
                           "B": [0.0, 0.10, -0.05, np.nan],     # B sem dados → proxy β·SPY
                           "SPY": [0.0, -0.02, -0.04, 0.01]}, index=idx)
    w = pd.Series([0.6, 0.4], index=["A", "B"])
    out = pf.stress_test(simple, w, pd.Series({"A": 1.0, "B": 2.0}),
                         {"teste": (str(idx[0].date()), str(idx[-1].date()))}, "SPY", cvar_now=0.02)
    s = out.set_index("serie")
    perda_a = 1 - 0.9 * 0.8 * 1.05
    perda_b = 1 - 0.96 * 0.92 * 1.02                              # proxy: 2 × SPY
    assert s.loc["A", "perda"] == pytest.approx(perda_a)
    assert bool(s.loc["B", "proxy"]) and s.loc["B", "perda"] == pytest.approx(perda_b)
    assert s.loc["Carteira", "perda"] == pytest.approx(0.6 * perda_a + 0.4 * perda_b)
    assert s.loc["Carteira", "perda"] == pytest.approx(s.loc[["A", "B"], "contrib"].sum())
    assert s.loc["A", "max_dd"] == pytest.approx(1 - 0.9 * 0.8)   # pico no início, vale no dia 3
    assert s.loc["A", "pior_dia"] == pytest.approx(0.20)
    # pior dia da carteira (buy-and-hold): dia 3, valor de 0,6·0,9+0,4·0,96 para 0,6·0,72+0,4·0,8832
    v2, v3 = 0.6 * 0.9 + 0.4 * 0.96, 0.6 * 0.72 + 0.4 * 0.8832
    assert s.loc["Carteira", "pior_dia"] == pytest.approx(1 - v3 / v2)
    assert s.loc["Carteira", "pior_dia_em_cvar"] == pytest.approx((1 - v3 / v2) / 0.02)
