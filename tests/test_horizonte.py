"""Horizontes de 10 e 21 dias: a recursão do GARCH à mão, o caso iid (raiz do tempo) e a ligação ao modelo de 1 dia."""
import numpy as np
import pandas as pd
import pytest

from config.settings import BT_WINDOW
from src.models import backtest as bt
from src.models import horizonte as hz


def linha(**kw):
    return pd.Series({"mu": 0.0, "sigma": 1.0, "omega": 0.0, "alpha": 0.0, "gamma": 0.0, "beta": 0.0} | kw)


def test_recursao_igual_ao_calculo_a_mao():
    """Resíduo sempre −1: σ² segue ω + (α + γ)ε² + βσ² e a perda é −(exp(Σr) − 1)."""
    g = linha(mu=0.05, sigma=2.0, omega=0.1, alpha=0.1, gamma=0.2, beta=0.7)
    perdas = hz.simulate_horizon(g, np.array([-1.0]), 3, 4, np.random.default_rng(0), method="fhs")
    s2, tot = 4.0, 0.0
    for _ in range(3):
        e = -np.sqrt(s2)
        tot += 0.05 + e
        s2 = 0.1 + (0.1 + 0.2) * e ** 2 + 0.7 * s2
    assert perdas == pytest.approx(np.full(4, -np.expm1(tot / 100)))


def test_sem_garch_vale_a_raiz_do_tempo():
    """Volatilidade constante e resíduos normais: o VaR a h dias é √h vezes o de 1 dia (em log, sem média)."""
    g = linha(sigma=1.0, omega=1.0)                               # σ² = ω todos os dias
    pool = np.random.default_rng(1).standard_normal(50_000)
    v1 = np.quantile(hz.simulate_horizon(g, pool, 1, 200_000, np.random.default_rng(2), "fhs"), 0.99)
    v10 = np.quantile(hz.simulate_horizon(g, pool, 10, 200_000, np.random.default_rng(3), "fhs"), 0.99)
    assert np.log1p(-v10) / np.log1p(-v1) == pytest.approx(np.sqrt(10), rel=0.03)


def test_volatilidade_alta_hoje_regressa_a_media():
    """Com σ de hoje acima da média de longo prazo, a simulação a 10 dias fica abaixo da raiz do tempo."""
    g = linha(sigma=3.0, omega=0.1, alpha=0.05, beta=0.85)         # variância de longo prazo = 0,1 / 0,10 = 1
    pool = np.random.default_rng(4).standard_normal(50_000)
    v1 = np.quantile(hz.simulate_horizon(g, pool, 1, 100_000, np.random.default_rng(5), "fhs"), 0.99)
    v10 = np.quantile(hz.simulate_horizon(g, pool, 10, 100_000, np.random.default_rng(6), "fhs"), 0.99)
    assert v10 < 0.9 * v1 * np.sqrt(10)


@pytest.fixture(scope="module")
def serie():
    rng = np.random.default_rng(7)
    n = BT_WINDOW + 130
    return pd.Series(rng.standard_t(6, n) * 0.01, index=pd.bdate_range("2022-01-03", periods=n), name="X")


def test_a_um_dia_bate_com_o_modelo_oficial(serie):
    """h = 1 é o modelo oficial: o VaR e o ES simulados batem com os de `next_day_forecast` (fórmula fechada)."""
    oficial = bt.next_day_forecast(serie.tail(BT_WINDOW + 45))
    g, pools = bt.garch_filter(bt.extend_next_day(serie.tail(BT_WINDOW + 45)), BT_WINDOW)
    m = hz.risk_measures(hz.simulate_horizon(g.iloc[-1], pools[int(g["refit"].iloc[-1])], 1, 400_000,
                                             np.random.default_rng(8)))
    for k in ("var99", "var975", "es975"):
        assert m[k] == pytest.approx(oficial[k], rel=0.04)
    prev = hz.forecast(serie, horizons=(10,), n_sim=20_000)
    assert prev["es975"].iloc[0] > prev["var975"].iloc[0] > m["var975"]          # 10 dias perde mais do que 1
    assert prev["var99_raiz"].iloc[0] == pytest.approx(oficial["var99"] * np.sqrt(10), rel=0.08)


def test_backtest_em_blocos_sem_sobreposicao(serie):
    res = hz.backtest_horizon(serie, 10, n_sim=500).set_index(["metodo", "nivel"])
    assert (res["blocos"] == 13).all()                               # 130 dias com previsão / 10
    assert res.loc[("simulacao", 0.99), "esperadas"] == pytest.approx(0.13)
    assert (res["excecoes"] <= res["blocos"]).all() and res["p_kupiec"].between(0, 1).all()
