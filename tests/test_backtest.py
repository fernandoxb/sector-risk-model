"""Testes do backtesting do VaR (src/models/backtest.py).

Correr: .venv/bin/python -m pytest
"""
import math

import numpy as np
import pandas as pd
import pytest
from arch import arch_model

from scipy.stats import norm
from scipy.stats import t as student_t

from src.collect.sessao import proximo_dia_util
from src.models import backtest as bt
from src.models.cockpit import cvar
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


# ----------------------------------------------------------------------------- ES histórico
def test_es_historico_usa_o_cvar_do_cockpit(rets):
    """ES previsto para o dia t = cockpit.cvar (fonte única) sobre t-window..t-1."""
    fc = bt.hist_forecast(rets, (0.975,), WINDOW)
    for t in [WINDOW, WINDOW + 40, len(rets) - 1]:
        assert fc["es_historico_0.975"].iloc[t] == pytest.approx(cvar(rets.iloc[t - WINDOW:t], 0.975))
        assert fc["var_historico_0.975"].iloc[t] == pytest.approx(hist_var(rets.iloc[t - WINDOW:t], 0.975))


# ----------------------------------------------------------------------------- GARCH e FHS
@pytest.fixture(scope="module")
def filtro(rets):
    return bt.garch_filter(rets, WINDOW, "M")


def _estimacao(rets, i):
    x = np.log1p(rets) * 100
    res = arch_model(x, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t") \
        .fit(first_obs=i - WINDOW, last_obs=i, disp="off")
    return x, res


def test_garch_igual_a_recursao_a_mao(rets, filtro):
    """VaR GARCH do backtest = recursão GJR escrita à mão com os parâmetros estimados até t-1."""
    lv = 0.99
    g, pools = filtro
    risk = bt.garch_risk(g, pools, (lv,))
    i = bt.refit_positions(rets.index, WINDOW, "M")[0]   # 1.ª estimação: mesmo ponto de partida do otimizador
    x, res = _estimacao(rets, i)
    mu, w, a, gm, b, nu = res.params[["mu", "omega", "alpha[1]", "gamma[1]", "beta[1]", "nu"]]
    cv = res.conditional_volatility.dropna()
    assert cv.index[-1] == x.index[i - 1]                 # estimação termina em t-1
    q = float(res.model.distribution.ppf(1 - lv, np.array([nu])))
    s2 = cv.iloc[-1] ** 2
    for d in (i, i + 1):                                  # previsões para os dias i e i+1
        e = x.iloc[d - 1] - mu
        s2 = w + (a + gm * (e < 0)) * e ** 2 + b * s2
        assert g["sigma"].iloc[d] == pytest.approx(math.sqrt(s2), rel=1e-6)
        assert risk["var_garch_0.99"].iloc[d] == pytest.approx(-math.expm1((mu + math.sqrt(s2) * q) / 100),
                                                               rel=1e-6)


@pytest.mark.parametrize("nu,lv", [(5.0, 0.975), (8.0, 0.99), (30.0, 0.975)])
def test_es_da_t_contra_formula_fechada(nu, lv):
    """Média da grelha da cauda = ES da t padronizada pela fórmula fechada:
    E[T | T <= q] = -f(q)/p · (ν + q²)/(ν - 1), reescalada por sqrt((ν-2)/ν)."""
    p = 1 - lv
    q = student_t.ppf(p, nu)
    fechada = -student_t.pdf(q, nu) / p * (nu + q ** 2) / (nu - 1) * math.sqrt((nu - 2) / nu)
    assert bt.t_tail_grid(lv, nu).mean() == pytest.approx(fechada, rel=2e-3)


def test_fhs_usa_residuos_da_janela_de_estimacao(rets, filtro):
    g, pools = filtro
    i = bt.refit_positions(rets.index, WINDOW, "M")[0]
    x, res = _estimacao(rets, i)
    z_mao = ((x - res.params["mu"]) / res.conditional_volatility).iloc[i - WINDOW:i].to_numpy()
    np.testing.assert_allclose(pools[i], z_mao, rtol=1e-9)
    risk = bt.garch_risk(g, pools, (0.975,))
    qe = np.quantile(z_mao, 0.025)
    assert g["refit"].iloc[i] == i                         # dia i usa a estimação feita em i
    mu, sg = g["mu"].iloc[i], g["sigma"].iloc[i]
    assert risk["var_fhs_0.975"].iloc[i] == pytest.approx(-math.expm1((mu + sg * qe) / 100))
    cauda = z_mao[z_mao <= qe]
    assert risk["es_fhs_0.975"].iloc[i] == pytest.approx(np.mean(-np.expm1((mu + sg * cauda) / 100)))


# ----------------------------------------------------------------------------- Z2 (Acerbi–Székely)
def test_z2_a_mao():
    loss = np.array([0.01, 0.05, -0.02, 0.04, 0.00])
    var, es = np.full(5, 0.03), np.full(5, 0.045)
    # exceções nos dias 2 e 4: Z2 = 1 - (0,05/0,045 + 0,04/0,045) / (5 · 0,2)
    assert bt.z2(loss, var, es, 0.2) == pytest.approx(1 - (0.05 / 0.045 + 0.04 / 0.045) / 1.0)


def test_z2_calibrado_sob_h0_e_negativo_se_es_subestimado():
    rng = np.random.default_rng(0)
    lv, sd = 0.975, 0.01
    p = 1 - lv
    var = np.full(1000, -norm.ppf(p) * sd)
    es = np.full(1000, norm.pdf(norm.ppf(p)) / p * sd)        # ES normal fechado
    certo = bt.z2(rng.normal(0, sd, (2000, 1000)), var, es, p)
    assert abs(certo.mean()) < 0.02
    errado = bt.z2(rng.normal(0, 1.5 * sd, (1, 1000)), var, es, p)[0]
    assert errado < -1 and bt.z2_pvalue(errado, certo) < 0.01


def test_simulacao_historica_so_usa_a_janela(rets):
    fc, aux, pools = bt.sector_forecasts(rets, (0.975,), WINDOW, "M")
    sims = bt.simulate_losses("historico", rets, aux, pools, WINDOW, 200, np.random.default_rng(1))
    j = 10
    pos = int(aux["pos"].iloc[j])
    janela = set(-rets.iloc[pos - WINDOW:pos].to_numpy())
    assert set(sims[:, j]) <= janela


# ----------------------------------------------------------------------------- FRTB
@pytest.mark.parametrize("n,lv,res", [(12, 0.99, "aprovada"), (13, 0.99, "reprovada"),
                                      (30, 0.975, "aprovada"), (31, 0.975, "reprovada")])
def test_frtb_por_mesa(n, lv, res):
    assert bt.frtb_desk(n, lv) == res


# ----------------------------------------------------------------------------- sem look-ahead
@pytest.mark.parametrize("metodo", ["historico", "garch", "fhs", "evt"])
def test_sem_look_ahead(rets, metodo):
    """Mudar o retorno do dia t (e de todos os seguintes) não pode alterar o VaR nem o ES de nenhum dia <= t."""
    fc0, _, _ = bt.sector_forecasts(rets, (0.99, 0.975), WINDOW, "M")
    pos = bt.refit_positions(rets.index, WINDOW, "M")
    cols = [c for c in fc0.columns if c.startswith((f"var_{metodo}", f"es_{metodo}"))]
    for t in [pos[2], pos[2] + 5]:       # um dia de re-estimação e um dia a meio do mês
        choque = rets.copy()
        choque.iloc[t:] = choque.iloc[t:] * 3 - 0.05
        fc1, _, _ = bt.sector_forecasts(choque, (0.99, 0.975), WINDOW, "M")
        dia = rets.index[t]
        pd.testing.assert_frame_equal(fc0.loc[:dia, cols], fc1.loc[:dia, cols])
        assert not np.allclose(fc0.loc[dia:, cols].iloc[1:5], fc1.loc[dia:, cols].iloc[1:5])


def test_tabela_tem_todas_as_combinacoes(rets):
    fc, aux, pools = bt.sector_forecasts(rets, (0.99, 0.975), WINDOW, "M")
    tab = bt.backtest_table(fc, aux, pools, rets, (0.99, 0.975), window=WINDOW, n_sim=200)
    assert len(tab) == 2 * len(bt.BT_METHODS) and set(tab["semaforo"]) <= {"verde", "amarelo", "vermelho"}
    assert (tab["dias"] == len(fc)).all() and tab["p_z2"].between(0, 1).all()


# ----------------------------------------------------------------------------- previsão oficial de amanhã
def test_previsao_de_amanha_igual_ao_backtest(rets):
    """A previsão para o próximo dia útil, feita com dados até t-1, coincide com o que o backtest
    produz para o dia t (sem look-ahead): mesmo código, mesma informação."""
    completo, _, _ = bt.sector_forecasts(rets, (0.99, 0.975), WINDOW, "M")
    pos = bt.refit_positions(rets.index, WINDOW, "M")
    for k in [pos[2], pos[2] + 7, len(rets) - 1]:          # dia de re-estimação, meio do mês, último dia
        prev = bt.next_day_forecast(rets.iloc[:k], "fhs", (0.99, 0.975), WINDOW, "M")
        dia = rets.index[k]
        assert prev["data_previsao"] == str(dia.date())
        assert prev["dados_ate"] == str(rets.index[k - 1].date())
        assert prev["var99"] == pytest.approx(completo.loc[dia, "var_fhs_0.99"], rel=1e-12)
        assert prev["var975"] == pytest.approx(completo.loc[dia, "var_fhs_0.975"], rel=1e-12)
        assert prev["es975"] == pytest.approx(completo.loc[dia, "es_fhs_0.975"], rel=1e-12)


def test_marcador_de_amanha_nao_altera_o_backtest(rets):
    """O dia extra (retorno 0) não muda nenhuma previsão anterior."""
    a, aux_a, _ = bt.sector_forecasts(rets, (0.99, 0.975), WINDOW, "M")
    b, aux_b, _ = bt.sector_forecasts(bt.extend_next_day(rets), (0.99, 0.975), WINDOW, "M")
    assert b.index[-1] == pd.Timestamp(proximo_dia_util(rets.index[-1].date()))
    pd.testing.assert_frame_equal(a, b.iloc[:-1], check_freq=False)   # só o atributo freq do índice difere
    pd.testing.assert_frame_equal(aux_a, aux_b.iloc[:-1], check_freq=False)


def test_previsao_usa_o_metodo_oficial(rets):
    from config.settings import OFFICIAL_VAR_METHOD
    prev = bt.next_day_forecast(rets, levels=(0.99, 0.975), window=WINDOW, refit_freq="M")
    assert prev["metodo"] == OFFICIAL_VAR_METHOD
    assert 0 < prev["var975"] < prev["var99"] and prev["es975"] > prev["var975"]


# ----------------------------------------------------------------------------- escolha fora da amostra e testes múltiplos
def test_holm_igual_ao_statsmodels():
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(3)
    p = np.concatenate([rng.uniform(0, 1, 40), rng.uniform(0, 0.002, 8)])
    np.testing.assert_array_equal(bt.holm(p, 0.05), multipletests(p, alpha=0.05, method="holm")[0])
    assert bt.holm([0.04, 0.03, 0.02], 0.05).sum() == 0          # 0,02 > 0,05/3: nenhuma sobrevive
    assert bt.holm([0.001, 0.02, 0.9], 0.05).tolist() == [True, True, False]


def _res(linhas):
    return pd.DataFrame(linhas, columns=["setor", "metodo", "nivel", "p_kupiec", "p_christoffersen", "p_cc", "p_z2"])


def test_escolha_do_metodo_e_contagens():
    res = _res([("A", "fhs", 0.975, .5, .5, .5, .5), ("B", "fhs", 0.975, .5, .01, .5, .5),
                ("A", "hist", 0.975, .5, .5, .5, .01), ("B", "hist", 0.975, .01, .5, .5, .5),
                ("A", "fhs", 0.99, .5, .5, .5, .5), ("B", "fhs", 0.99, .5, .5, .5, .5),
                ("A", "hist", 0.99, .5, .5, .5, .5), ("B", "hist", 0.99, .5, .5, .5, .5)])
    p = bt.passes(res).set_index(["metodo", "nivel"])
    assert p.loc[("fhs", 0.975), ["p_kupiec", "p_christoffersen", "p_z2", "tudo", "series"]].tolist() == [2, 1, 2, 1, 2]
    assert p.loc[("hist", 0.975), "tudo"] == 0
    assert bt.choose_method(res, level=0.975) == "fhs"
    empate = res.copy()
    empate.loc[(empate.metodo == "hist") & (empate.nivel == 0.975), ["p_kupiec", "p_z2"]] = 0.5   # hist: 2 a 97,5%
    empate.loc[(empate.metodo == "fhs") & (empate.setor == "B") & (empate.nivel == 0.975), "p_christoffersen"] = 0.5
    empate.loc[(empate.metodo == "hist") & (empate.setor == "A") & (empate.nivel == 0.99), "p_kupiec"] = 0.01
    assert bt.choose_method(empate, level=0.975) == "fhs"        # empate a 97,5%: decide o outro nível
    m = bt.multiple_testing(res).set_index("metodo")
    assert m.loc["fhs", ["testes", "rejeicoes"]].tolist() == [16, 1] and m.loc["fhs", "esperadas_por_acaso"] == 0.8


def test_periodos_usam_as_mesmas_previsoes(rets):
    """Com split, 'total' é igual ao run sem split e as exceções dos dois períodos somam as do total."""
    simple = rets.to_frame("X")
    w = bt.BT_WINDOW                                   # run usa a janela do projeto: há len(rets) − w previsões
    meio = str(rets.index[w + (len(rets) - w) // 2].date())
    base, _, _ = bt.run(simple, ["X"])
    res, _, _ = bt.run(simple, ["X"], split=meio, min_days=10)
    total = res[res.periodo == "total"].drop(columns="periodo").reset_index(drop=True)
    pd.testing.assert_frame_equal(total, base)
    chave = ["metodo", "nivel"]
    soma = (res[res.periodo != "total"].groupby(chave)["excecoes"].sum()
            .reindex(pd.MultiIndex.from_frame(base[chave])).to_numpy())
    np.testing.assert_array_equal(soma, base["excecoes"].to_numpy())
    assert set(res.periodo) == {"total", "selecao", "confirmacao"}
    curto, _, _ = bt.run(simple, ["X"], split=meio, min_days=10_000)          # períodos curtos demais ficam de fora
    assert set(curto.periodo) == {"total"}


# ----------------------------------------------------------------------------- EVT condicional (cauda de Pareto)
def test_formulas_da_cauda_iguais_ao_scipy():
    from scipy.stats import genpareto
    u, xi, beta, frac = 1.3, 0.2, 0.6, 0.10
    for p in (0.05, 0.025, 0.01, 0.001):
        ref = u + genpareto.ppf(1 - p / frac, xi, scale=beta)        # P(X > x) = frac · (1 − G(x − u))
        assert bt.gpd_quantile(p, u, xi, beta, frac) == pytest.approx(ref, rel=1e-12)
    # ES = média dos quantis da cauda (integração numérica fina) = fórmula fechada 7.19
    grelha = bt.gpd_quantile(0.025 * (np.arange(200_000) + 0.5) / 200_000, u, xi, beta, frac)
    assert bt.gpd_es(0.025, u, xi, beta, frac) == pytest.approx(grelha.mean(), rel=1e-3)
    assert bt.gpd_quantile(0.01, u, 0.0, beta, frac) == pytest.approx(u - beta * np.log(0.1))   # limite exponencial
    assert bt.gpd_quantile(frac, u, xi, beta, frac) == pytest.approx(u)                           # no limiar


def test_ajuste_da_cauda_recupera_quantis_extremos():
    from scipy.stats import t as t_dist
    rng = np.random.default_rng(12)
    z = rng.standard_t(4, 20_000)                                    # perdas = −z, cauda pesada (ξ teórico = 1/4)
    u, xi, beta, frac = bt.gpd_tail(z)
    assert frac == pytest.approx(0.10, abs=1e-4) and 0.05 < xi < 0.45 and beta > 0
    for p in (0.01, 0.001):                                          # extrapola para lá do que a amostra mostra bem
        assert bt.gpd_quantile(p, u, xi, beta, frac) == pytest.approx(t_dist.ppf(1 - p, 4), rel=0.10)
    _, xi_n, _, _ = bt.gpd_tail(rng.normal(size=20_000))             # cauda leve: ξ perto de zero ou negativo
    assert xi_n < 0.08 < xi
    assert bt.gpd_tail(z) == bt.gpd_tail(z.copy())                   # mesmo conjunto de resíduos: mesmo ajuste (cache)


def test_evt_na_previsao_e_na_simulacao(rets):
    fc, aux, pools = bt.sector_forecasts(rets, (0.99, 0.975), WINDOW, "M")
    i = int(aux["refit"].iloc[-1])
    gp = bt.gpd_tail(pools[i])
    mu, sig = aux["mu"].iloc[-1], aux["sigma"].iloc[-1]
    assert fc["var_evt_0.99"].iloc[-1] == pytest.approx(float(bt._to_loss(mu, sig, -bt.gpd_quantile(0.01, *gp))))
    assert (fc["es_evt_0.975"] > fc["var_evt_0.975"]).all() and (fc["var_evt_0.99"] > fc["var_evt_0.975"]).all()
    # o corpo da distribuição é o da FHS: abaixo do limiar os dois VaR ficam próximos
    assert fc["var_evt_0.975"].corr(fc["var_fhs_0.975"]) > 0.95
    sim = bt.simulate_losses("evt", rets, aux, pools, WINDOW, 400, np.random.default_rng(0))
    assert sim.shape == (400, len(aux))
    limiar = bt._to_loss(aux["mu"].to_numpy(), aux["sigma"].to_numpy(), -gp[0])
    ultimos = aux["refit"].to_numpy() == i
    assert (sim[:, ultimos] > limiar[ultimos]).mean() == pytest.approx(gp[3], abs=0.02)   # fração na cauda


# ----------------------------------------------------------------------------- teste de McNeil–Frey ao ES
def test_mcneil_frey_deteta_es_subestimado():
    rng = np.random.default_rng(31)
    n, p = 6000, 0.025
    from scipy.stats import t as t_dist
    loss = rng.standard_t(5, n) * 0.01
    var = np.full(n, t_dist.ppf(1 - p, 5) * 0.01)
    es_certo = np.full(n, loss[loss > var[0]].mean())
    ok = bt.mcneil_frey(loss, var, es_certo)
    assert ok["n"] == int((loss > var).sum()) and abs(ok["media"]) < 1e-12 and ok["pvalue"] > 0.3
    mau = bt.mcneil_frey(loss, var, es_certo * 0.75)                 # ES 25% abaixo do real
    assert mau["media"] > 0.2 and mau["pvalue"] < 0.01
    assert bt.mcneil_frey(loss, var, es_certo * 1.3)["pvalue"] > 0.9  # ES a mais não é rejeitado (teste unilateral)
    assert np.isnan(bt.mcneil_frey(loss[:50], var[:50], es_certo[:50])["pvalue"])   # poucas exceções
