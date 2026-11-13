"""Testes da dependência de cauda (src/models/cauda.py), com dados simulados de parâmetros conhecidos."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.models import cauda as cd


def amostra_t(n, rho, nu, seed):
    """Vetor bivariado t(ν) com correlação ρ: normal correlacionada a dividir por √(χ²_ν / ν)."""
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    return pd.DataFrame(z / np.sqrt(rng.chisquare(nu, n) / nu)[:, None], columns=["A", "B"])


# ----------------------------------------------------------------------------- fórmula do λ (tabela 5.1 do livro)
@pytest.mark.parametrize("nu,rho,lam", [(2, -0.5, 0.06), (2, 0, 0.18), (2, 0.5, 0.39), (2, 0.9, 0.72),
                                        (4, -0.5, 0.01), (4, 0, 0.08), (4, 0.5, 0.25), (4, 0.9, 0.63),
                                        (10, 0, 0.01), (10, 0.5, 0.08), (10, 0.9, 0.46)])
def test_lambda_igual_a_tabela_do_livro(nu, rho, lam):
    assert cd.lambda_t(rho, nu) == pytest.approx(lam, abs=0.005)


def test_lambda_limites():
    assert cd.lambda_t(1.0, 4) == pytest.approx(1.0)                 # dependência perfeita
    assert cd.lambda_t(0.9, 1e6) < 1e-6                              # ν → ∞: cópula de Gauss, sem dependência de cauda
    assert cd.lambda_t(0.5, 3) > cd.lambda_t(0.5, 8) > cd.lambda_t(0.2, 8)


# ----------------------------------------------------------------------------- cópula t
def test_tau_de_kendall_recupera_a_correlacao():
    x = amostra_t(6000, 0.6, 5, seed=1)
    assert cd.tau_to_rho(x.corr(method="kendall").loc["A", "B"]) == pytest.approx(0.6, abs=0.03)
    assert cd.tau_to_rho(0.0) == 0 and cd.tau_to_rho(1.0) == pytest.approx(1.0)


def test_verosimilhanca_igual_a_scipy():
    x = amostra_t(300, 0.5, 6, seed=2)
    u = cd.pseudo_obs(x).to_numpy()
    p = np.array([[1, 0.5], [0.5, 1]])
    z = stats.t.ppf(u, 6)
    ref = stats.multivariate_t(shape=p, df=6).logpdf(z).sum() - stats.t.logpdf(z, 6).sum()
    assert cd.t_copula_loglik(u, p, 6) == pytest.approx(ref, rel=1e-10)


def test_ajuste_distingue_cauda_pesada_de_gauss():
    pesada, gauss = amostra_t(4000, 0.5, 4, seed=3), amostra_t(4000, 0.5, 1e7, seed=4)
    f_p, f_g = (cd.fit_group(x, x) for x in (pesada, gauss))
    assert f_p["nu"] == pytest.approx(4, abs=1.2) and f_g["nu"] > 25
    assert f_p["lam"].loc["A", "B"] == pytest.approx(cd.lambda_t(0.5, 4), abs=0.07)
    assert f_g["lam"].loc["A", "B"] < 0.03 < f_p["lam"].loc["A", "B"]
    assert f_p["rho_t"].loc["A", "B"] == pytest.approx(0.5, abs=0.04)


def test_correcao_da_matriz():
    ruim = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])       # não é definida positiva
    assert np.linalg.eigvalsh(ruim).min() < 0
    boa = cd.nearest_corr(ruim)
    assert np.linalg.eigvalsh(boa).min() > 0 and np.allclose(np.diag(boa), 1) and np.allclose(boa, boa.T)
    ok = np.array([[1, 0.3], [0.3, 1]])
    assert np.array_equal(cd.nearest_corr(ok), ok)


# ----------------------------------------------------------------------------- quedas conjuntas e grafo
def test_quedas_conjuntas():
    rng = np.random.default_rng(7)
    n = 4000
    a = rng.normal(size=n)
    x = pd.DataFrame({"A": a, "IGUAL": a, "INDEP": rng.normal(size=n), "OPOSTA": -a})
    p = cd.joint_crash(x, 0.05)
    assert p.loc["A", "IGUAL"] == 1 and p.loc["A", "A"] == 1
    assert p.loc["A", "INDEP"] == pytest.approx(0.05, abs=0.04)            # independência ≈ q
    assert p.loc["A", "OPOSTA"] == 0
    assert np.allclose(p, p.T)                                               # simétrica: as duas caudas têm prob. q
    t = cd.joint_crash(amostra_t(20000, 0.5, 3, seed=8), 0.01)               # cauda pesada: bem acima de q
    assert t.loc["A", "B"] > 0.25


def test_tabela_de_pares_e_grafo():
    rng = np.random.default_rng(9)
    f = rng.normal(size=1500)
    x = pd.DataFrame({"A": f + 0.3 * rng.normal(size=1500), "B": f + 0.3 * rng.normal(size=1500),
                      "C": rng.normal(size=1500), "D": rng.normal(size=1500)})
    fit = cd.fit_group(x, x)
    pares = cd.pairs_table(fit, "g")
    assert len(pares) == 6 and set(pares.columns) == {"grupo", "a", "b", "corr", "tau", "rho_t", "lam", "p_conj"}
    forte = pares.sort_values("p_conj").iloc[-1]
    assert {forte.a, forte.b} == {"A", "B"}
    g = cd.graph(pares, by="p_conj")
    assert g.number_of_edges() == 3 and g.has_edge("A", "B")                # árvore com N−1 arestas, com a mais forte
    assert set(g["A"]["B"]) == {"weight", "corr", "lam", "p_conj"}


# ----------------------------------------------------------------------------- cenários conjuntos
def test_simulacao_da_copula_tem_margens_uniformes_e_a_correlacao_pedida():
    p = np.array([[1.0, 0.6], [0.6, 1.0]])
    u = cd.simulate_t_copula(p, 5.0, n=200_000, seed=1)
    assert u.mean(0) == pytest.approx([0.5, 0.5], abs=0.005) and u.std(0) == pytest.approx(12 ** -0.5, abs=0.005)
    tau = stats.kendalltau(u[:5000, 0], u[:5000, 1])[0]
    assert np.sin(np.pi * tau / 2) == pytest.approx(0.6, abs=0.04)          # inverte tau_to_rho


def test_cenario_igual_a_normal_bivariada_quando_nu_e_grande():
    """Com ν muito alto a cópula t é a de Gauss: P(B na cauda | A na cauda) vem da normal bivariada do scipy."""
    rho, q = 0.7, 0.05
    p = np.array([[1.0, rho], [rho, 1.0]])
    rets = pd.DataFrame(np.random.default_rng(2).normal(0, 0.01, (750, 2)), columns=["A", "B"])
    c = cd.scenarios(rets, p, nu=1e6, q=q, n=400_000, seed=3).set_index(["cenario", "serie"])
    z = stats.norm.ppf(q)
    exato = stats.multivariate_normal([0, 0], p).cdf([z, z]) / q
    assert c.loc[("A", "B"), "p_cauda"] == pytest.approx(exato, abs=0.01)
    assert c.loc[("A", "A"), "p_cauda"] == 1.0
    assert c.loc[("A", "A"), "perda_media"] == pytest.approx(-rets["A"][rets["A"] <= rets["A"].quantile(q)].mean(), rel=0.05)
    assert c.loc[("A", "B"), "perda_media"] > c.loc[("A", "B"), "perda_normal"]      # dia mau de A arrasta B
    assert c.loc[("A", "B"), "p_obs"] == cd.joint_crash(rets, q).loc["A", "B"]       # contagem nos dias reais
    piores = rets["A"].nsmallest(int(q * len(rets))).index
    assert c.loc[("A", "B"), "perda_obs"] == pytest.approx(-rets.loc[piores, "B"].mean())


def test_cauda_pesada_da_mais_quedas_conjuntas_do_que_gauss():
    p = np.array([[1.0, 0.5], [0.5, 1.0]])
    rets = pd.DataFrame(np.random.default_rng(4).normal(0, 0.01, (750, 2)), columns=["A", "B"])
    pc = [cd.scenarios(rets, p, nu, q=0.01, n=400_000, seed=5).set_index(["cenario", "serie"]).loc[("A", "B"), "p_cauda"]
          for nu in (3.0, 1e6)]
    assert pc[0] > pc[1] + 0.1
    assert pc[0] > cd.lambda_t(p, 3.0)[0, 1] - 0.02       # λ é o limite quando q → 0; a 1% ainda fica por cima



# ----------------------------------------------------------------------------- janelas
def test_janelas_usam_os_anos_pedidos_e_a_cobertura():
    """Na janela de 10 anos entra uma série com 8 anos (cobertura 80%) e as contas usam os dias comuns; uma série
    com 2 anos fica de fora. Com 1 ano e q = 10% há ~25 dias de cauda por série."""
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2014-01-01", "2025-12-31")
    comum = rng.standard_t(5, len(idx)) * 0.01
    p = pd.DataFrame({c: 100 * np.exp(np.cumsum(comum + rng.normal(0, 0.006, len(idx)))) for c in "ABCD"}, index=idx)
    p.loc[: idx[-1] - pd.DateOffset(years=8), "C"] = np.nan          # C só tem os últimos 8 anos
    p.loc[: idx[-1] - pd.DateOffset(years=2), "D"] = np.nan          # D só tem os últimos 2
    f10 = cd._fit(p, list("ABCD"), anos=10, q=0.05)
    assert list(f10["corr"].columns) == ["A", "B", "C"]
    assert f10["n"] == p["C"].notna().sum() - 1                       # dias comuns: começam com C
    f1 = cd._fit(p, list("ABCD"), anos=1, q=0.10)
    assert list(f1["corr"].columns) == list("ABCD") and 250 < f1["n"] < 265
    assert f1["p_conj"].loc["A", "B"] > 0.3                           # A e B partilham o choque comum
