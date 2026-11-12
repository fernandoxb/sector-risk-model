"""Testes do modelo por empresa e da previsão da carteira (src/models/backtest_empresas.py), com dados sintéticos."""
import numpy as np
import pandas as pd
import pytest

from src.models import backtest as bt
from src.models import backtest_empresas as be

W = 300


def test_empresas_com_historico():
    idx = pd.bdate_range("2020-01-01", periods=100)
    p = pd.DataFrame({"LONGA": 1.0, "CURTA": [np.nan] * 60 + [1.0] * 40}, index=idx)
    assert be.com_historico(p, min_days=50) == ["LONGA"]


def test_resumo_e_validado():
    res = pd.DataFrame({
        "empresa": ["A", "B", "A", "B"], "metodo": ["fhs", "fhs", "historico", "historico"], "nivel": 0.975,
        "p_kupiec": [0.5, 0.5, 0.5, 0.01], "p_christoffersen": [0.5, 0.2, 0.01, 0.01], "p_cc": [0.5, 0.3, 0.01, 0.01],
        "p_z2": [0.5, 0.01, 0.5, 0.5], "z2": [0.0, -0.4, 0.1, 0.1]})
    r = be.resumo_metodos(res).set_index("metodo")
    assert r.loc["fhs", ["Kupiec", "Christoffersen", "Z2 (ES)", "series"]].tolist() == [2, 2, 1, 2]
    assert r.loc["historico", ["Kupiec", "Christoffersen", "Z2 (ES)"]].tolist() == [1, 0, 2]
    v = be.validado(res, 0.975)
    assert v.to_dict() == {"A": True, "B": False}                 # B falha o Z2


def test_previsao_da_carteira_e_o_modelo_sobre_os_retornos_da_carteira():
    rng = np.random.default_rng(5)
    n = W + 80
    idx = pd.bdate_range("2023-01-02", periods=n)
    rets = pd.DataFrame(rng.standard_t(6, (n, 2)) * 0.012, columns=["A", "B"], index=idx)
    w = pd.Series({"A": 0.7, "B": 0.3})
    prev = be.portfolio_forecast(rets, w, window=W)
    direto = bt.next_day_forecast((rets @ w).tail(W + 45), window=W)
    assert prev["es975"] == pytest.approx(direto["es975"]) and prev["var99"] == pytest.approx(direto["var99"])
    assert prev["es975"] > prev["var975"] > 0 and prev["var99"] > prev["var975"] and prev["dias"] == n
    # diversificação: o ES da carteira é menor do que a soma ponderada dos ES isolados
    isolados = sum(w[c] * bt.next_day_forecast(rets[c].tail(W + 45), window=W)["es975"] for c in w.index)
    assert prev["es975"] < isolados
    # uma posição com pouco histórico tira dias comuns: sem janela, não há previsão
    curta = rets.copy()
    curta.loc[curta.index[:200], "B"] = np.nan
    assert be.portfolio_forecast(curta, w, window=W) is None


# ----------------------------------------------------------------------------- correção do ES
def test_multiplicador_poe_o_z2_a_zero():
    rng = np.random.default_rng(9)
    n, p = 2000, 0.025
    loss = rng.standard_t(4, n) * 0.02                      # caudas pesadas
    var = np.full(n, np.quantile(loss, 1 - p))
    es = np.full(n, loss[loss > var[0]].mean() / 1.4)       # ES previsto 1,4× abaixo do real
    z = float(bt.z2(loss, var, es, p))
    assert z < -0.3
    k = be.es_multiplier(z, p_z2=0.001, cap=10)
    assert k == pytest.approx(1 - z)
    assert float(bt.z2(loss, var, es * k, p)) == pytest.approx(0, abs=1e-12)     # com o ES corrigido, Z2 = 0
    assert be.es_multiplier(z, p_z2=0.001, cap=1.5) == min(1 - z, 1.5)           # limitado
    assert be.es_multiplier(-0.4, p_z2=0.20) == 1.0                               # Z2 não rejeita: sem correção
    assert be.es_multiplier(+0.3, p_z2=0.001) == 1.0                              # nunca reduz o ES


def test_marcas_e_es_corrigido():
    res = pd.DataFrame({
        "empresa": ["A", "B"] * 2, "metodo": "fhs", "nivel": [0.975, 0.975, 0.99, 0.99],
        "p_kupiec": 0.5, "p_christoffersen": 0.5, "p_z2": [0.5, 0.01, 0.5, 0.5], "z2": [0.0, -0.3, 0.0, 0.0],
        "semaforo": ["verde", "verde", "verde", "amarelo"]})
    fl = be.flags(res)
    assert fl.loc["B", ["semaforo", "validado", "es_subestimado"]].tolist() == ["amarelo", False, True]
    assert fl.loc["A", "mult_es"] == 1.0 and fl.loc["B", "mult_es"] == pytest.approx(1.3)
    off = be.with_flags(pd.DataFrame({"empresa": ["A", "B", "NOVA"], "es975": [0.04, 0.05, 0.06]}), fl)
    assert off["es975_aj"].tolist() == pytest.approx([0.04, 0.065, 0.06])          # NOVA: sem backtest, sem correção


def test_previsao_diaria_igual_a_do_backtest_completo():
    rng = np.random.default_rng(21)
    n = W + 200
    idx = pd.bdate_range("2023-01-02", periods=n)
    prices = pd.DataFrame({"X": 100 * np.cumprod(1 + rng.standard_t(6, n) * 0.012)}, index=idx)
    rapida = be.daily_forecast(prices, ["X"], window=W).iloc[0]
    completa = bt.next_day_forecast(prices["X"].pct_change().dropna(), window=W)   # com a história toda
    for c in ("var99", "var975", "es975"):
        assert rapida[c] == pytest.approx(completa[c], rel=0.01)                   # só muda o ponto de partida do GARCH
    assert rapida["data_previsao"] == completa["data_previsao"]


def test_quando_corre_o_backtest_completo():
    ant = pd.DataFrame({"empresa": ["A", "B"]})
    assert be.needs_full("2026-11-11", ["A", "B"], None)                     # sem backtest anterior
    assert be.needs_full("2026-11-11", ["A", "C"], ant)                      # as empresas mudaram
    assert not be.needs_full("2026-11-11", ["A", "B"], ant)                  # quarta: só previsão
    assert be.needs_full("2026-11-13", ["A", "B"], ant)                      # sexta: completo
    assert be.needs_full("2026-04-02", ["A", "B"], ant)                      # quinta antes da Sexta-feira Santa
    assert be.needs_full("2026-11-11", ["A", "B"], ant, weekly=False)


def test_backtest_da_carteira():
    rng = np.random.default_rng(4)
    n = 900
    idx = pd.bdate_range("2023-01-02", periods=n)
    rets = pd.DataFrame(rng.standard_t(6, (n, 2)) * 0.012, columns=["A", "B"], index=idx)
    w = pd.Series({"A": 0.6, "B": 0.4})
    b = be.portfolio_backtest(rets, w)
    t = b["testes"].set_index("nivel")
    assert b["dias"] == t["dias"].iloc[0] and 100 < b["dias"] < n - 756 + 1
    s = b["serie"]
    assert int((-s["ret"] > s["var"]).sum()) == int(t.loc[0.975, "excecoes"])    # a série bate com a tabela
    pv = b["previsao"]
    assert pv["es975_aj"] == pytest.approx(pv["es975"] * pv["mult_es"]) and pv["mult_es"] >= 1
    assert pv["es975"] == pytest.approx(be.portfolio_forecast(rets, w)["es975"], rel=0.01)
    assert be.portfolio_backtest(rets.tail(780), w) is None                     # sem 3 meses de previsões
