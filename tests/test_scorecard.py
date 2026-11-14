"""Testes do scorecard (src/models/scorecard.py).

Correr: .venv/bin/python -m pytest
"""
import numpy as np
import pandas as pd
import pytest

from config.settings import SCORE_IMPACT, SCORE_PROBABILITY, SCORE_WEIGHTS
from src.models import scorecard as sc

TICK = [f"S{i}" for i in range(11)]


@pytest.fixture
def raw() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "cauda": rng.uniform(0.02, 0.08, 11), "sistemico": rng.uniform(0.5, 1.5, 11),
        "stress": rng.uniform(0.1, 0.6, 11), "regime": rng.uniform(0, 1, 11),
        "momento": rng.integers(0, 5, 11).astype(float),
    }, index=TICK)


@pytest.fixture
def verde() -> pd.Series:
    return pd.Series("verde", index=TICK)


# ----------------------------------------------------------------------------- normalização
def test_percentil_extremos_e_empates():
    s = pd.Series([3.0, 1.0, 2.0, 2.0, 5.0])
    np.testing.assert_allclose(sc.percentile_norm(s), [0.75, 0.0, 0.375, 0.375, 1.0])


def test_normalizacao_em_0_1_e_regime_intacto(raw):
    c = sc.normalize(raw)
    assert ((c >= 0) & (c <= 1)).all().all()
    pd.testing.assert_series_equal(c["regime"], raw["regime"])
    assert c["cauda"].min() == 0 and c["cauda"].max() == 1


# ----------------------------------------------------------------------------- pesos
def test_pesos_somam_1_e_grupos_cobrem_tudo():
    assert sum(SCORE_WEIGHTS.values()) == pytest.approx(1)
    assert set(SCORE_IMPACT) | set(SCORE_PROBABILITY) == set(SCORE_WEIGHTS)
    assert not set(SCORE_IMPACT) & set(SCORE_PROBABILITY)


def test_pesos_que_nao_somam_1_sao_recusados(raw, verde):
    with pytest.raises(ValueError):
        sc.scorecard(raw, verde, {k: 0.3 for k in SCORE_WEIGHTS})


def test_score_a_mao(raw, verde):
    out = sc.scorecard(raw, verde)
    c = sc.normalize(raw)
    t = TICK[4]
    mao = 100 * sum(SCORE_WEIGHTS[k] * c.loc[t, k] for k in SCORE_WEIGHTS)
    assert out.loc[t, "score"] == pytest.approx(mao)
    assert out["score"].between(0, 100).all()


# ----------------------------------------------------------------------------- monotonia
@pytest.mark.parametrize("comp", list(SCORE_WEIGHTS))
def test_piorar_um_componente_nunca_baixa_o_score(raw, verde, comp):
    base = sc.scorecard(raw, verde)
    for t in TICK:
        pior = raw.copy()
        pior.loc[t, comp] = pior.loc[t, comp] * 1.5 + (0.5 if comp == "regime" else 1)
        if comp == "regime":
            pior.loc[t, comp] = min(pior.loc[t, comp], 1.0)
        novo = sc.scorecard(pior, verde)
        assert novo.loc[t, "score"] >= base.loc[t, "score"] - 1e-12
        assert novo.loc[t, "impacto"] >= base.loc[t, "impacto"] or comp in SCORE_PROBABILITY
        assert novo.loc[t, "probabilidade"] >= base.loc[t, "probabilidade"] or comp in SCORE_IMPACT


def test_semaforo_pior_nunca_baixa_o_score_final(raw, verde):
    base = sc.scorecard(raw, verde)
    for cor, m in [("amarelo", 1.10), ("vermelho", 1.25)]:
        sem = verde.copy()
        sem[TICK[0]] = cor
        out = sc.scorecard(raw, sem)
        assert out.loc[TICK[0], "score_final"] == pytest.approx(min(100, base.loc[TICK[0], "score"] * m))
        assert bool(out.loc[TICK[0], "risco_modelo"]) and not out.loc[TICK[1], "risco_modelo"]


# ----------------------------------------------------------------------------- stress e momento
def test_proxy_pesa_metade():
    stress = pd.DataFrame({
        "serie": ["A", "A", "A", "B", "B"],
        "perda": [0.40, 0.20, 0.10, 0.30, 0.30],
        "proxy": [True, False, False, False, False],
    })
    out = sc.stress_loss(stress, ["A", "B"], 0.5)
    assert out["A"] == pytest.approx((0.5 * 0.40 + 0.20 + 0.10) / 2.5)
    assert out["B"] == pytest.approx(0.30)
    assert sc.stress_loss(stress, ["A"], 1.0)["A"] == pytest.approx(0.70 / 3)


def test_stress_only_losses_truncates_gains():
    """Um ganho num cenário (XLE em 2022) conta como zero: não anula as perdas dos outros cenários."""
    stress = pd.DataFrame({
        "serie": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
        "perda": [0.50, 0.40, -0.445, 0.50, 0.40, 0.0, 0.50, 0.40, 0.10],
        "proxy": False,
    })
    out = sc.stress_loss(stress, ["A", "B", "C"], 0.5, only_losses=True)
    assert out["A"] == pytest.approx(out["B"]) == pytest.approx(0.30)
    assert out["C"] > out["A"]                                              # mais perda, mais stress
    assert sc.stress_loss(stress, ["A"], 0.5, only_losses=False)["A"] == pytest.approx(0.455 / 3)


def test_momento_conta_so_os_ultimos_dias():
    d = pd.bdate_range("2026-01-01", periods=80)
    ret = np.zeros(80)
    ret[[5, 25, 30, 70, 75]] = -0.05                  # perda 5% > VaR 3%: 5 exceções no total
    series = pd.DataFrame({"data": d, "setor": "A", "ret": ret, "var_fhs_0.975": 0.03})
    # últimos 60 dias = índices 20..79 → conta 25, 30, 70, 75 (o dia 5 fica de fora)
    assert sc.momentum(series, ["A"], "fhs", 0.975, 60)["A"] == 4
    assert sc.momentum(series, ["A"], "fhs", 0.975, 10)["A"] == 2


# ----------------------------------------------------------------------------- sensibilidade
def test_sensibilidade(raw, verde):
    s = sc.sensitivity(raw, verde, factor=0.5)
    assert len(s) == 2 * len(SCORE_WEIGHTS)
    assert sc.sensitivity(raw, verde, factor=0.0)["mesma_ordem"].all()
    k = "cauda"
    novo = SCORE_WEIGHTS[k] * 1.5
    assert s[(s.componente == k) & (s.variacao == "+50%")]["peso_novo"].iloc[0] == \
        pytest.approx(novo / (1 + 0.5 * SCORE_WEIGHTS[k]))


# ----------------------------------------------------------------------------- história própria e incerteza do ranking
def test_percentil_na_historia_do_proprio_setor():
    series = pd.DataFrame({"setor": ["A"] * 4 + ["B"] * 4, "es_x_0.975": [0.01, 0.02, 0.03, 0.04] * 2})
    p = sc.own_history_pct(series, pd.Series({"A": 0.025, "B": 0.05}), ["A", "B"], method="x")
    assert p.to_dict() == {"A": 0.5, "B": 1.0}                 # metade da história de A fica abaixo do ES de hoje


def test_cauda_junta_posicao_e_historia(raw):
    base = sc.normalize(raw)
    com = sc.normalize(raw.assign(cauda_hist=1.0), hist_weight=0.5)
    assert list(com.columns) == list(sc.COMPONENTS)                              # cauda_hist não é um componente
    assert com["cauda"].to_numpy() == pytest.approx(0.5 * base["cauda"].to_numpy() + 0.5)
    assert com.drop(columns="cauda").equals(base.drop(columns="cauda"))
    # todos os setores no máximo da sua história: o menos arriscado dos 11 deixa de ter cauda 0
    assert com["cauda"].min() == pytest.approx(0.5)


def test_incerteza_do_ranking(raw, verde):
    dominante = raw.copy()
    dominante.loc[TICK[0]] = raw.max() * 2                      # o pior em tudo: 1.º com quaisquer pesos
    u = sc.rank_uncertainty(dominante, verde, n=500, seed=1)
    assert (u.loc[TICK[0], ["rank_p05", "rank_p95"]] == 1).all() and u.loc[TICK[0], "p_top3"] == 1.0
    assert (u["rank_p05"] <= u["rank_p95"]).all() and u["p_top3"].sum() == pytest.approx(3.0)
    fixo = sc.rank_uncertainty(raw, verde, n=200, conc=1e9, seed=2)             # pesos sem variar: sem incerteza
    oficial = sc.scorecard(raw, verde)["rank"]
    assert (fixo["rank_p05"] == fixo["rank_p95"]).all() and (fixo["rank_p05"] == oficial.reindex(fixo.index)).all()



def test_grupos_juntam_intervalos_que_se_sobrepoem():
    sc = pd.DataFrame({"score_final": [90, 70, 68, 60, 40, 39],
                       "rank_p05": [1, 2, 2, 3, 5, 5], "rank_p95": [1, 4, 3, 4, 6, 6]}, index=list("ABCDEF"))
    assert sc.pipe(sc_grupos).tolist() == [1, 2, 2, 2, 3, 3]
    sem = pd.DataFrame({"score_final": [2, 1], "rank": [1, 2]}, index=list("XY"))
    assert sem.pipe(sc_grupos).tolist() == [1, 2]                              # sem intervalos: a posição


def sc_grupos(df):
    return sc.grupos(df)


def test_grupos_exigem_sobreposicao_com_todos():
    """C toca em B mas não em A: com a regra 'todos', C abre um grupo novo (em cadeia ficaria com A e B)."""
    sc_ = pd.DataFrame({"score_final": [3, 2, 1], "rank_p05": [1, 1, 2], "rank_p95": [1, 2, 3]}, index=list("ABC"))
    assert sc_grupos(sc_).tolist() == [1, 1, 2]
