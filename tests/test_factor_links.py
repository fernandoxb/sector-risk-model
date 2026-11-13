"""Testes das ligações medidas fator → setor (src/models/factor_links.py).

Correr: .venv/bin/python -m pytest
"""
import numpy as np
import pandas as pd
import pytest

from src.models import factor_links as fl

N = 156   # 3 anos de semanas


@pytest.fixture(scope="module")
def dados():
    """Fatores semanais simulados e setores com betas conhecidos."""
    rng = np.random.default_rng(11)
    idx = pd.date_range("2023-01-06", periods=N, freq="W-FRI")
    f = pd.DataFrame({"a": rng.normal(0, 0.10, N), "b": rng.normal(0, 0.05, N)}, index=idx)
    ruido = rng.standard_t(5, (N, 3)) * 0.01
    r = pd.DataFrame({
        "POS": 0.20 * f["a"] + ruido[:, 0],      # beta +0,20 em a
        "NEG": -0.30 * f["a"] + ruido[:, 1],     # beta −0,30 em a
        "NADA": ruido[:, 2],                     # sem relação
    }, index=idx)
    return r, f


FAT = {"a": {"transf": "diff", "adverso": +1}, "b": {"transf": "diff", "adverso": -1}}


# ----------------------------------------------------------------------------- recuperar beta, sinal e significância
def test_recupera_beta_e_significancia(dados):
    r, f = dados
    pos, neg, nada = (fl.regress(r[c], f["a"]) for c in ("POS", "NEG", "NADA"))
    assert pos["beta"] == pytest.approx(0.20, abs=0.03) and pos["t"] > 2
    assert neg["beta"] == pytest.approx(-0.30, abs=0.03) and neg["t"] < -2
    assert abs(nada["t"]) < 2
    assert pos["n"] == N and pos["lags"] == fl.hac_lags(N)


def test_sinal_segue_a_direcao_adversa(dados):
    r, f = dados
    m = fl.link_matrix(r, f, FAT, t_min=2).set_index(["fator", "setor"])
    # a sobe = adverso: beta > 0 → o adverso melhora o setor → "−"; beta < 0 → piora → "+"
    assert m.loc[("a", "POS"), "sinal"] == "-" and m.loc[("a", "NEG"), "sinal"] == "+"
    assert not m.loc[("a", "NADA"), "aceite"] and pd.isna(m.loc[("a", "NADA"), "sinal"])
    # com a direção adversa invertida (descer é adverso), o sinal inverte
    inv = fl.link_matrix(r[["POS"]], f[["a"]], {"a": {"transf": "diff", "adverso": -1}}, t_min=2)
    assert inv["sinal"].iloc[0] == "+"


def test_intensidade_por_tercis(dados):
    r, f = dados
    m = fl.link_matrix(r, f, FAT, t_min=2)
    acc = m[m["aceite"]]
    assert set(acc["w"]) <= {1, 2, 3} and m.loc[~m["aceite"], "w"].isna().all()
    # a ligação mais forte tem a intensidade máxima
    assert acc.loc[acc["beta_pad"].abs().idxmax(), "w"] == acc["w"].max()


# ----------------------------------------------------------------------------- HAC à mão
def newey_west_a_mao(y: np.ndarray, x: np.ndarray, lags: int) -> float:
    """Erro-padrão HAC do declive: (X'X)⁻¹ S (X'X)⁻¹ com Bartlett w_l = 1 - l/(L+1).
    Newey–West clássico, sem correção n/(n-k) (a opção por omissão do statsmodels)."""
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.solve(X.T @ X, X.T @ y)
    u = (y - X @ b)[:, None] * X                   # scores x_t · e_t
    S = u.T @ u
    for lag in range(1, lags + 1):
        g = u[lag:].T @ u[:-lag]
        S += (1 - lag / (lags + 1)) * (g + g.T)
    inv = np.linalg.inv(X.T @ X)
    return float(np.sqrt((inv @ S @ inv)[1, 1]))


@pytest.mark.parametrize("lags", [0, 2, 4])
def test_hac_igual_ao_calculo_a_mao(dados, lags):
    r, f = dados
    res = fl.regress(r["POS"], f["a"], lags)
    assert res["se"] == pytest.approx(newey_west_a_mao(r["POS"].to_numpy(), f["a"].to_numpy(), lags), rel=1e-10)


def test_regra_de_lags():
    assert fl.hac_lags(156) == 4 and fl.hac_lags(100) == 4 and fl.hac_lags(500) == 5


# ----------------------------------------------------------------------------- dados semanais
def test_corte_na_ultima_sexta():
    assert fl.weekly_cut(pd.Timestamp("2026-11-09")) == pd.Timestamp("2026-11-06")   # segunda → sexta anterior
    assert fl.weekly_cut(pd.Timestamp("2026-11-06")) == pd.Timestamp("2026-11-06")   # sexta fica


def test_transformacoes():
    lv = pd.DataFrame({"j": [4.0, 4.25, 4.10], "p": [50.0, -1.0, 55.0]},
                      index=pd.date_range("2026-01-02", periods=3, freq="W-FRI"))
    ch = fl.factor_changes(lv, {"j": {"transf": "diff"}, "p": {"transf": "logdiff"}})
    np.testing.assert_allclose(ch["j"].iloc[1:], [0.25, -0.15])
    assert ch["p"].iloc[1:].isna().all()          # nível <= 0 → sem log


def test_semana_usa_o_ultimo_valor():
    d = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.to_datetime(["2026-11-02", "2026-11-06", "2026-11-09", "2026-11-13"]))
    w = fl.weekly_levels(d.to_frame("x"))
    assert w["x"].tolist() == [2.0, 4.0]          # sexta 06 e sexta 13



# ----------------------------------------------------------------------------- regressão conjunta
@pytest.fixture(scope="module")
def colineares():
    """A causa o setor; B anda com A (r ≈ 0,85) mas não tem efeito próprio."""
    rng = np.random.default_rng(3)
    n = 2000
    idx = pd.date_range("2000-01-07", periods=n, freq="W-FRI")
    a = rng.normal(size=n)
    b = 0.85 * a + np.sqrt(1 - 0.85 ** 2) * rng.normal(size=n)
    f = pd.DataFrame({"a": a, "b": b}, index=idx)
    r = pd.DataFrame({"S": 0.02 * a + rng.normal(0, 0.02, n)}, index=idx)
    return r, f


def test_multivariate_ridge_separates_collinear_factors(colineares):
    r, f = colineares
    assert f["a"].corr(f["b"]) == pytest.approx(0.85, abs=0.02)
    uni = fl.link_matrix(r, f, FAT, t_min=2).set_index("fator")
    assert uni.loc["b", "aceite"]                                    # um fator de cada vez: B é aceite por engano
    multi = fl.fit_multivariate_ridge(r, f, alpha=1.0, beta_min=0.05, factors=FAT, t_min=2).set_index("fator")
    assert multi.loc["a", "aceite_multi"] and not multi.loc["b", "aceite_multi"]
    assert abs(multi.loc["b", "beta_pad_multi"]) < 0.05 < abs(multi.loc["a", "beta_pad_multi"])
    assert multi.loc["a", "b_multi"] == pytest.approx(0.02, abs=0.003)          # volta à escala original
    assert multi.loc["a", "sinal_multi"] == fl.MINUS and pd.isna(multi.loc["b", "sinal_multi"])


def test_ridge_igual_ao_sklearn_e_alinha_datas(colineares):
    r, f = colineares
    f2 = f.iloc[:-1].copy()                              # o FRED vem uma semana atrás
    f2.iloc[5, 0] = np.nan
    multi = fl.fit_multivariate_ridge(r, f2, alpha=7.0, factors=FAT)
    assert (multi["n_multi"] == len(r) - 2).all()
    Ridge = pytest.importorskip("sklearn.linear_model").Ridge
    d = r.join(f2, how="inner").dropna()
    z = (d - d.mean()) / d.std()
    coef = Ridge(alpha=7.0, fit_intercept=True).fit(z[["a", "b"]], z["S"]).coef_
    assert multi["beta_pad_multi"].to_numpy() == pytest.approx(coef, rel=1e-9)
