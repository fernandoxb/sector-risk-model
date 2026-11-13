"""Testes da carteira do utilizador (src/models/carteira.py), com dados sintéticos."""
import numpy as np
import pandas as pd
import pytest

from src.models import carteira as ca


def test_texto_do_link():
    u = {"AAPL", "MSFT", "XLU"}
    c = ca.parse_carteira("AAPL:30, msft:20;XLU:5%, AAPL:10, LIXO:5, MSFT:-3, x, MSFT:abc", u)
    assert c == {"AAPL": 40.0, "MSFT": 20.0, "XLU": 5.0}
    assert ca.parse_carteira(None) == {} and ca.parse_carteira("AAPL:1,5", None) == {"AAPL": 1.0}
    assert ca.format_carteira({"AAPL": 0.3, "MSFT": 0.2}) == "AAPL:60,MSFT:40"
    assert ca.parse_carteira(ca.format_carteira(c)) == pytest.approx({"AAPL": 61.5, "MSFT": 30.8, "XLU": 7.7})
    muitas = ",".join(f"T{i}:1" for i in range(40))
    assert len(ca.parse_carteira(muitas)) == ca.USER_PORTFOLIO_MAX


def test_normalizar():
    w = ca.normalizar({"A": 30, "B": 10, "C": 0})
    assert w.to_dict() == {"A": 0.75, "B": 0.25}
    with pytest.raises(ValueError):
        ca.normalizar({"A": 0})


def test_exposicao_setorial():
    h = pd.DataFrame({"setor": ["XLK", "XLK", "XLC", "XLY"], "ticker": ["AAPL", "MSFT", "GOOGL", "GOOGL"],
                      "nome": "x", "peso": [0.15, 0.12, 0.10, 0.02]})
    m = ca.mapa_setores(h)
    assert m["GOOGL"] == "XLC" and m["XLU"] == "XLU" and m["SPY"] == "Mercado"
    e = ca.exposicao_setorial(ca.normalizar({"AAPL": 40, "MSFT": 10, "XLU": 30, "SPY": 20}), m)
    assert e["XLK"] == pytest.approx(0.5) and e["XLU"] == pytest.approx(0.3) and e["Mercado"] == pytest.approx(0.2)
    assert e["XLF"] == 0 and e.sum() == pytest.approx(1) and len(e) == 12


def test_risco_soma_e_beta():
    rng = np.random.default_rng(3)
    n = 600
    mkt = pd.Series(rng.normal(0, 0.01, n), index=pd.bdate_range("2024-01-01", periods=n))
    rets = pd.DataFrame({"A": 1.5 * mkt + rng.normal(0, 0.005, n), "B": 0.5 * mkt + rng.normal(0, 0.005, n)},
                        index=mkt.index)
    w = ca.normalizar({"A": 60, "B": 40})
    tab, res = ca.risco_carteira(rets, w, mkt)
    assert tab["contrib_cvar"].sum() == pytest.approx(res["cvar"]) and tab["pct_cvar"].sum() == pytest.approx(1)
    assert res["beta"] == pytest.approx(0.6 * tab.loc["A", "beta"] + 0.4 * tab.loc["B", "beta"])   # beta é linear
    assert res["beta"] == pytest.approx(1.1, abs=0.08)
    assert tab.loc["A", "racio"] > 1 > tab.loc["B", "racio"]              # A pesa mais no risco do que no capital
    assert res["posicoes"] == 2 and res["maior_peso"] == pytest.approx(0.6)


def test_stress_e_score():
    st = pd.DataFrame({"cenario": ["X"] * 3 + ["Y"] * 3, "serie": ["XLK", "XLU", "SPY"] * 2,
                       "perda": [0.5, 0.2, 0.4, 0.3, -0.1, 0.2]})
    e = pd.Series({"XLK": 0.5, "XLU": 0.3, "Mercado": 0.2})
    s = ca.stress_setorial(e, st).set_index("cenario")
    assert s.loc["X", "perda"] == pytest.approx(0.5 * 0.5 + 0.3 * 0.2 + 0.2 * 0.4)
    assert s.loc["Y", "perda"] == pytest.approx(0.5 * 0.3 - 0.3 * 0.1 + 0.2 * 0.2)
    assert s.loc["X", "maior_contribuinte"] == "XLK" and s.loc["X", "perda_mercado"] == 0.4
    assert ca.score_carteira(e, pd.Series({"XLK": 90.0, "XLU": 20.0})) == pytest.approx((0.5 * 90 + 0.3 * 20) / 0.8)


# ----------------------------------------------------------------------------- HRP
@pytest.fixture(scope="module")
def tres_ativos():
    """A e B voláteis e a cair juntos (o mesmo choque comum); C independente e calmo."""
    rng = np.random.default_rng(21)
    n = 800
    comum = rng.standard_t(4, n) * 0.02
    return pd.DataFrame({"A": comum + rng.normal(0, 0.0005, n), "B": comum + rng.normal(0, 0.0005, n),
                         "C": rng.normal(0, 0.006, n)}, index=pd.bdate_range("2023-01-02", periods=n))


@pytest.mark.parametrize("medida", ["cvar", "variance"])
def test_hrp_weights_sum_to_one_and_non_negative(tres_ativos, medida):
    rng = np.random.default_rng(22)
    r = tres_ativos.assign(D=rng.normal(0, 0.01, 800), E=rng.normal(0, 0.015, 800))
    w = ca.calcular_pesos_hrp(r, risk_measure=medida)
    assert list(w.index) == list(r.columns)
    assert w.sum() == pytest.approx(1.0, abs=1e-10) and (w >= 0).all()


def test_hrp_penalizes_tail_correlated_cluster(tres_ativos):
    from src.models.cauda import joint_crash
    assert joint_crash(tres_ativos).loc["A", "B"] >= 0.9                 # A e B caem juntos
    w = ca.calcular_pesos_hrp(tres_ativos)
    assert w["C"] > w["A"] + w["B"]
    # dois ativos: é a paridade de risco pelo CVaR, calculada à mão
    par = ca.calcular_pesos_hrp(tres_ativos[["A", "C"]])
    ca_, cc = ck_cvar(tres_ativos["A"]), ck_cvar(tres_ativos["C"])
    assert par["A"] == pytest.approx(1 - ca_ / (ca_ + cc))


def test_hrp_single_asset_edge_case(tres_ativos):
    assert ca.calcular_pesos_hrp(tres_ativos[["A"]]).to_dict() == {"A": 1.0}
    assert ca.calcular_pesos_hrp(tres_ativos.tail(100)) is None           # poucos dias comuns
    d = pd.DataFrame(1.0, index=list("ABC"), columns=list("ABC"))         # distância dada de fora, com NaN
    d.loc["A", "B"] = d.loc["B", "A"] = np.nan
    assert ca.calcular_pesos_hrp(tres_ativos, matriz_distancia=d).sum() == pytest.approx(1.0)


def ck_cvar(r):
    from config.settings import CVAR_LEVEL
    from src.models.cockpit import cvar
    return cvar(r, CVAR_LEVEL)

