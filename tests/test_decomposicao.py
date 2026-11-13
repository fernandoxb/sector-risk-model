"""Decomposição mercado / setor / específico: soma 1 e bate com uma regressão independente (statsmodels)."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.models import decomposicao as dc


@pytest.fixture(scope="module")
def rets():
    rng = np.random.default_rng(8)
    n = 600
    m = rng.normal(0, 0.010, n)
    s = 0.9 * m + rng.normal(0, 0.008, n)                 # o setor anda com o mercado
    e = 1.2 * m + 0.7 * (s - 0.9 * m) + rng.normal(0, 0.012, n)
    return pd.DataFrame({"SPY": m, "XLA": s, "EMP": e, "SOLO": rng.normal(0, 0.02, n)},
                        index=pd.bdate_range("2024-01-01", periods=n))


def test_partes_somam_um_e_batem_com_o_ols(rets):
    d = dc.decompose(rets["EMP"], rets["SPY"], rets["XLA"])
    assert d["pct_mercado"] + d["pct_setor"] + d["pct_especifico"] == pytest.approx(1)
    assert d["pct_mercado"] == pytest.approx(rets["EMP"].corr(rets["SPY"]) ** 2)
    ols = sm.OLS(rets["EMP"], sm.add_constant(rets[["SPY", "XLA"]])).fit()
    assert d["pct_mercado"] + d["pct_setor"] == pytest.approx(ols.rsquared)
    assert d["beta_setor"] == pytest.approx(ols.params["XLA"])            # o mesmo coeficiente do setor
    assert d["beta_mercado"] == pytest.approx(1.2, abs=0.1) and d["beta_setor"] == pytest.approx(0.7, abs=0.15)


def test_empresa_independente_e_quase_toda_risco_proprio(rets):
    assert dc.decompose(rets["SOLO"], rets["SPY"], rets["XLA"])["pct_especifico"] > 0.97


def test_sem_historico_nao_ha_decomposicao(rets):
    assert dc.decompose(rets["EMP"].tail(100), rets["SPY"], rets["XLA"]) is None
    hold = pd.DataFrame({"setor": ["XLA", "XLA", "XLZ"], "ticker": ["EMP", "FALTA", "SOLO"]})
    t = dc.table(rets, hold, market="SPY")
    assert t["empresa"].tolist() == ["EMP"]               # sem preços ou sem setor: fica de fora
