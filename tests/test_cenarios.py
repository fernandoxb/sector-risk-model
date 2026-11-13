"""Motor de cenários: recupera betas conhecidos, a normal condicional bate com a fórmula e a validação não espreita
o futuro."""
import numpy as np
import pandas as pd
import pytest

from src.models import cenarios as cn

F = cn.FAT


@pytest.fixture(scope="module")
def dados():
    """Fatores semanais com juros e mercado correlacionados; setores com betas conhecidos; uma empresa por setor."""
    rng = np.random.default_rng(3)
    n = 1500
    idx = pd.date_range("1995-01-06", periods=n, freq="W-FRI")
    x = pd.DataFrame(rng.normal(0, 0.1, (n, len(F))), columns=F, index=idx)
    x["mercado"] = 0.6 * x["juros"] / 0.1 * 0.02 + 0.8 * rng.normal(0, 0.02, n)         # correlação 0,6 com os juros
    d = x.copy()
    d["XLU"] = -0.05 * x["juros"] + 0.5 * x["mercado"] + rng.normal(0, 0.005, n)
    d["XLE"] = 0.10 * x["petroleo"] + 1.0 * x["mercado"] + rng.normal(0, 0.005, n)
    d["U1"] = 1.5 * d["XLU"] + rng.normal(0, 0.004, n)
    hold = pd.DataFrame({"setor": ["XLU"], "ticker": ["U1"], "nome": ["U1"], "peso": [1.0]})
    return d, hold


def test_recupera_os_betas(dados):
    d, hold = dados
    mod = cn.ajustar(d, hold, setores=["XLU", "XLE"])
    B = pd.DataFrame(mod["B"], index=mod["setores"], columns=F)
    assert B.loc["XLU", "juros"] == pytest.approx(-0.05, abs=0.005) and B.loc["XLU", "mercado"] == pytest.approx(0.5, abs=0.02)
    assert B.loc["XLE", "petroleo"] == pytest.approx(0.10, abs=0.005) and abs(B.loc["XLE", "juros"]) < 0.005
    assert mod["beta"][0] == pytest.approx(1.5, abs=0.02)


def test_normal_condicional_igual_a_formula(dados):
    d, hold = dados
    mod = cn.ajustar(d, hold, setores=["XLU", "XLE"])
    s = cn.simular(mod, {"juros": 0.3}, h=4, n=40_000, seed=1)
    S, mu = mod["sigma"] * 4, mod["mu"] * 4
    i, j = F.index("mercado"), F.index("juros")
    esperado = mu[i] + S[i, j] / S[j, j] * (0.3 - mu[j])                          # E[mercado | juros = 0,3]
    assert s["mercado"].mean() == pytest.approx(esperado, abs=0.003)
    assert (s["juros"] == 0.3).all()
    # o setor recebe o choque direto e o do mercado que o acompanha; a empresa, 1,5 vezes o do setor
    xlu = np.log1p(s["XLU"]).mean()
    assert xlu == pytest.approx(4 * mod["a"][0] + (mod["B"][0] * s[F].mean().to_numpy()).sum(), abs=0.002)
    assert np.log1p(s["U1"]).mean() == pytest.approx(4 * mod["alfa"][0] + 1.5 * xlu, abs=0.003)


def test_sem_choque_e_a_distribuicao_normal_dos_fatores(dados):
    d, hold = dados
    mod = cn.ajustar(d, hold, setores=["XLU"])
    s = cn.simular(mod, {}, h=1, n=40_000, seed=2)
    assert s[F].std().to_numpy() == pytest.approx(np.sqrt(np.diag(mod["sigma"])), rel=0.03)


def test_episodio_nao_espreita_o_futuro(dados):
    d, hold = dados
    ini, fim = "2010-01-01", "2010-03-05"
    base = cn.episodio(d, hold, ini, fim, anos=10, n=2000)
    depois = d.copy()
    depois.loc[depois.index > pd.Timestamp(fim), :] *= 5                          # o que vem depois não conta
    assert cn.episodio(depois, hold, ini, fim, anos=10, n=2000).equals(base)
    dentro = d.copy()
    dentro.loc[(dentro.index > pd.Timestamp(ini)) & (dentro.index <= pd.Timestamp(fim)), "XLU"] -= 0.05
    v = cn.episodio(dentro, hold, ini, fim, anos=10, n=2000).set_index("setor")
    assert v.loc["XLU", "real"] < base.set_index("setor").loc["XLU", "real"]      # o real muda, a previsão não
    assert v.loc["XLU", "mediana"] == pytest.approx(base.set_index("setor").loc["XLU", "mediana"])
    assert base["semanas"].iloc[0] == 9
