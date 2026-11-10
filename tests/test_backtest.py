"""Testes do backtesting do VaR (src/models/backtest.py).

Correr: .venv/bin/python -m pytest
"""
import math

import numpy as np
import pandas as pd
import pytest
from arch import arch_model

from src.models import backtest as bt
from src.models.risk_metrics import hist_var

WINDOW = 300   # janela curta para os testes serem rápidos


@pytest.fixture(scope="module")
def rets() -> pd.Series:
    """Retornos simulados com clusters de volatilidade (GARCH) e caudas t."""
    rng = np.random.default_rng(42)
    n = 800
    z = rng.standard_t(5, n) / np.sqrt(5 / 3)
    s2, r = np.empty(n), np.empty(n)
    s2[0] = 1e-4
    for t in range(n):
        if t:
            s2[t] = 2e-6 + 0.08 * r[t - 1] ** 2 + 0.9 * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    return pd.Series(r, index=pd.bdate_range("2020-01-01", periods=n), name="X")


# ----------------------------------------------------------------------------- Kupiec
def kupiec_a_mao(x: int, t: int, p: float) -> float:
    """LR_pof escrito diretamente com math.log, sem scipy."""
    ph = x / t
    l0 = (t - x) * math.log(1 - p) + x * math.log(p)
    l1 = (t - x) * math.log(1 - ph) + (x * math.log(ph) if x else 0.0)
    return -2 * (l0 - l1)


@pytest.mark.parametrize("x,t,p", [(8, 250, 0.01), (2, 250, 0.01), (15, 250, 0.025), (60, 4741, 0.01)])
def test_kupiec_contra_calculo_a_mao(x, t, p):
    lr = kupiec_a_mao(x, t, p)
    res = bt.kupiec(x, t, p)
    assert res["lr"] == pytest.approx(lr, rel=1e-12)
    # p-valor de χ²(1) = erfc(sqrt(LR/2)), também sem scipy
    assert res["pvalue"] == pytest.approx(math.erfc(math.sqrt(lr / 2)), rel=1e-9)


def test_kupiec_valor_de_referencia():
    # 8 exceções em 250 dias a 99%: LR = 7,7336 (rejeita a 5%, χ²(1) crítico = 3,841)
    res = bt.kupiec(8, 250, 0.01)
    assert res["lr"] == pytest.approx(7.7336, abs=1e-4)
    assert res["pvalue"] < 0.01


def test_kupiec_sem_excecoes_e_taxa_exata():
    assert bt.kupiec(0, 250, 0.01)["lr"] == pytest.approx(-2 * 250 * math.log(0.99))
    assert bt.kupiec(10, 1000, 0.01)["lr"] == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------------- Christoffersen
def test_christoffersen_contra_calculo_a_mao():
    hits = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    n00, n01, n10, n11 = bt.transition_counts(hits)
    assert (n00, n01, n10, n11) == (10, 3, 3, 3)
    pi0, pi1, pi = 3 / 13, 3 / 6, 6 / 19
    l0 = 13 * math.log(1 - pi) + 6 * math.log(pi)
    l1 = 10 * math.log(1 - pi0) + 3 * math.log(pi0) + 3 * math.log(1 - pi1) + 3 * math.log(pi1)
    assert bt.christoffersen(hits)["lr"] == pytest.approx(-2 * (l0 - l1), rel=1e-12)


def test_christoffersen_sem_excecoes_consecutivas():
    hits = np.zeros(500, dtype=int)
    hits[::50] = 1                  # exceções isoladas: π1 = 0 < π0, LR finito
    res = bt.christoffersen(hits)
    assert res["n11"] == 0 and np.isfinite(res["lr"])


def test_condicional_e_a_soma():
    hits = np.array([0] * 90 + [1, 1, 1] + [0] * 107)
    cc = bt.conditional_coverage(hits, 0.01)
    soma = bt.kupiec(3, 200, 0.01)["lr"] + bt.christoffersen(hits)["lr"]
    assert cc["lr"] == pytest.approx(soma)
    assert cc["pvalue"] == pytest.approx(math.exp(-soma / 2))   # χ²(2): sf(x) = e^(-x/2)


# ----------------------------------------------------------------------------- semáforo
def test_zonas_de_basileia_a_99():
    assert bt.traffic_light_zones(0.99, 250) == (4, 9)


@pytest.mark.parametrize("n,zona", [(0, "verde"), (4, "verde"), (5, "amarelo"), (9, "amarelo"),
                                    (10, "vermelho")])
def test_semaforo(n, zona):
    hits = pd.Series([1] * n + [0] * (250 - n))
    assert bt.traffic_light(hits, 0.99) == (zona, n)


def test_semaforo_usa_so_ultimos_250():
    hits = pd.Series([1] * 50 + [0] * 250)
    assert bt.traffic_light(hits, 0.99) == ("verde", 0)


# ----------------------------------------------------------------------------- exceções
def test_excecao_e_perda_maior_que_var():
    idx = pd.bdate_range("2024-01-01", periods=4)
    r = pd.Series([-0.03, -0.02, 0.01, -0.0201], index=idx)
    var = pd.Series([0.02, 0.02, 0.02, np.nan], index=idx)
    h = bt.exceptions(r, var)
    assert h.tolist() == [1, 0, 0]          # perda igual ao VaR não é exceção; NaN fica de fora


# ----------------------------------------------------------------------------- VaR histórico
def test_var_historico_igual_ao_do_modelador(rets):
    """O VaR previsto para o dia t é o hist_var de risk_metrics sobre t-window..t-1."""
    v = bt.hist_var_forecast(rets, 0.99, WINDOW)
    for t in [WINDOW, WINDOW + 17, len(rets) - 1]:
        assert v.iloc[t] == pytest.approx(hist_var(rets.iloc[t - WINDOW:t], 0.99), rel=1e-12)
    assert v.iloc[:WINDOW].isna().all()


# ----------------------------------------------------------------------------- GARCH
def test_garch_igual_a_recursao_a_mao(rets):
    """VaR GARCH do backtest = recursão GJR escrita à mão com os parâmetros estimados até t-1."""
    lv = 0.99
    fc = bt.garch_var_forecast(rets, (lv,), WINDOW, "M")
    x = np.log1p(rets) * 100
    i = bt.refit_positions(rets.index, WINDOW, "M")[0]   # 1.ª estimação: mesmo ponto de partida do otimizador
    res = arch_model(x, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t") \
        .fit(first_obs=i - WINDOW, last_obs=i, disp="off")
    mu, w, a, g, b, nu = res.params[["mu", "omega", "alpha[1]", "gamma[1]", "beta[1]", "nu"]]
    # variância condicional do último dia da amostra de estimação, depois recursão para a frente
    cv = res.conditional_volatility.dropna()
    assert cv.index[-1] == x.index[i - 1]                 # estimação termina em t-1
    s2 = cv.iloc[-1] ** 2
    e = x.iloc[i - 1] - mu
    s2 = w + (a + g * (e < 0)) * e ** 2 + b * s2          # previsão para o dia i
    q = float(res.model.distribution.ppf(1 - lv, np.array([nu])))
    assert fc.iloc[i, 0] == pytest.approx(-math.expm1((mu + math.sqrt(s2) * q) / 100), rel=1e-6)
    e = x.iloc[i] - mu
    s2 = w + (a + g * (e < 0)) * e ** 2 + b * s2          # previsão para o dia i+1
    assert fc.iloc[i + 1, 0] == pytest.approx(-math.expm1((mu + math.sqrt(s2) * q) / 100), rel=1e-6)


# ----------------------------------------------------------------------------- sem look-ahead
@pytest.mark.parametrize("metodo", ["historico", "garch"])
def test_sem_look_ahead(rets, metodo):
    """Mudar o retorno do dia t (e de todos os seguintes) não pode alterar o VaR de nenhum dia <= t."""
    fc0 = bt.sector_forecasts(rets, (0.99, 0.975), WINDOW, "M")
    pos = bt.refit_positions(rets.index, WINDOW, "M")
    for t in [pos[2], pos[2] + 5]:       # um dia de re-estimação e um dia a meio do mês
        choque = rets.copy()
        choque.iloc[t:] = choque.iloc[t:] * 3 - 0.05
        fc1 = bt.sector_forecasts(choque, (0.99, 0.975), WINDOW, "M")
        dia = rets.index[t]
        cols = [c for c in fc0.columns if c.startswith(f"var_{metodo}")]
        pd.testing.assert_frame_equal(fc0.loc[:dia, cols], fc1.loc[:dia, cols])
        assert not np.allclose(fc0.loc[dia:, cols].iloc[1:5], fc1.loc[dia:, cols].iloc[1:5])


def test_tabela_tem_todas_as_combinacoes(rets):
    fc = bt.sector_forecasts(rets, (0.99, 0.975), WINDOW, "M")
    tab = bt.backtest_table(fc, (0.99, 0.975))
    assert len(tab) == 4 and set(tab["semaforo"]) <= {"verde", "amarelo", "vermelho"}
    assert (tab["dias"] == len(fc)).all()
