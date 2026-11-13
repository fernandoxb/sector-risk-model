"""Teia de risco (ECharts): nós e ligações coerentes, e a cadeia de um fator acende só o que lhe pertence."""
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "cockpit"))
from config.settings import MACRO_FACTORS, SECTORS  # noqa: E402
from paineis import teia  # noqa: E402

SET = list(SECTORS)


@pytest.fixture
def dados():
    sc = pd.DataFrame({"score_final": range(11), "bruto_cauda": [0.02 + 0.002 * i for i in range(11)]}, index=SET)
    links = pd.DataFrame([{"fator": "juros", "setor": "XLU", "aceite_multi": True, "beta_pad_multi": -0.3, "t_multi": -3.0,
                           "sinal_multi": "+"},
                          {"fator": "juros", "setor": "XLE", "aceite_multi": True, "beta_pad_multi": 0.2, "t_multi": 2.5,
                           "sinal_multi": "-"},
                          {"fator": "vix", "setor": "XLK", "aceite_multi": False, "beta_pad_multi": -0.01, "t_multi": -0.2,
                           "sinal_multi": None}])
    pares = pd.DataFrame([{"grupo": "setores", "a": a, "b": b, "p_conj": 0.1 + 0.01 * (i + j), "corr": 0.5, "lam": 0.1}
                          for i, a in enumerate(SET) for j, b in enumerate(SET) if i < j])
    hold = pd.DataFrame([{"setor": s, "ticker": f"{s}{k}", "nome": f"Empresa {k}", "peso": 0.1} for s in SET for k in range(3)])
    return sc, links, pares, hold


def test_nos_e_ligacoes_coerentes(dados):
    sc, links, pares, hold = dados
    op = teia.opcoes(sc, links, pares, hold, None, None)
    g = op["series"][0]
    ids = [n["id"] for n in g["data"]]
    assert len(ids) == len(set(ids)) == len(MACRO_FACTORS) + 11 + len(hold)
    assert all(a["source"] in ids and a["target"] in ids for a in g["links"])
    fs = [a for a in g["links"] if a["source"] in MACRO_FACTORS]
    assert {(a["source"], a["target"]) for a in fs} == {("juros", "XLU"), ("juros", "XLE")}   # só as aceites
    assert [a["lineStyle"]["type"] for a in fs] == ["solid", "dashed"]                      # agrava cheio, atenua tracejado
    assert sum(a["source"] in SET and a["target"] in SET for a in g["links"]) == 10           # árvore dos 11 setores
    x, y = zip(*(n["value"] for n in g["data"]))
    assert max(map(abs, x + y)) < teia.LIM


def test_cadeia_de_um_fator(dados):
    sc, links, pares, hold = dados
    op = teia.opcoes(sc, links, pares, hold, None, None, foco="juros")
    nos = {n["id"]: n for n in op["series"][0]["data"]}
    acesos = {i for i, n in nos.items() if n["itemStyle"]["opacity"] > teia.APAGADO}
    assert acesos == {"juros", "XLU", "XLE"} | {f"{s}/{s}{k}" for s in ("XLU", "XLE") for k in range(3)}
    vagas = op["series"][1:]
    assert len(vagas) == 2 and len(vagas[0]["data"]) == 2 and len(vagas[1]["data"]) == 6    # fator → setores → empresas
    assert vagas[1]["effect"]["delay"] > vagas[0]["effect"]["delay"]


def test_evento_do_clique():
    assert teia.evento({"chart_event": "XLK|123"}) == "XLK|123"
    assert teia.evento({}) is None and teia.evento(None) is None and teia.evento({"chart_event": 5}) is None


def test_carteira_e_correlacao(dados):
    sc, links, pares, hold = dados
    op = teia.opcoes(sc, links, pares, hold, None, None, carteira={"XLK0", "XLE"}, por="corr")
    nos = {n["id"]: n for n in op["series"][0]["data"]}
    assert nos["XLK/XLK0"]["itemStyle"]["borderWidth"] == 2 and nos["XLK/XLK0"]["label"]["show"]
    assert nos["XLK/XLK1"]["itemStyle"]["borderWidth"] == 0
    assert nos["XLE"]["itemStyle"]["borderWidth"] == 3
    cauda = [a for a in op["series"][0]["links"] if a["source"] in SET and a["target"] in SET]
    assert len(cauda) == 10 and all("correlação" in a["dica"] for a in cauda)


def test_carteira_no_cenario():
    import numpy as np
    from paineis.cenario import carteira_no_cenario
    sims = pd.DataFrame({"A": [0.10, -0.10], "B": [0.0, 0.2], "mercado": [np.log(1.05), np.log(0.95)]})
    r = carteira_no_cenario(sims, pd.Series({"A": 0.25, "SPY": 0.25, "FORA": 0.5}))
    assert r.tolist() == pytest.approx([0.5 * 0.10 + 0.5 * 0.05, 0.5 * -0.10 + 0.5 * -0.05])   # FORA sai, pesos re-normalizados
    assert carteira_no_cenario(sims, pd.Series({"FORA": 1.0})) is None


def test_acao_fora_do_mapa_na_teia(dados):
    sc, links, pares, hold = dados
    h = pd.concat([hold, pd.DataFrame({"setor": ["XLK"], "ticker": ["ASML"], "nome": ["ASML Holding"], "peso": [0.0]})])
    nos = {n["id"]: n for n in teia.opcoes(sc, links, pares, h, None, None, carteira={"ASML"})["series"][0]["data"]}
    assert "fora do mapa" in nos["XLK/ASML"]["dica"] and nos["XLK/ASML"]["label"]["show"]


def test_tickers_do_link():
    from paineis.minha_carteira import tickers_do_link
    assert tickers_do_link("aapl:30, ASML.AS:10;brk-b:5,lixo") == ["AAPL", "ASML.AS", "BRK-B"]


def test_numeros_do_cenario_na_teia(dados):
    """Com um cenário: o setor e as empresas da cadeia mostram a percentagem; a ligação do fator chocado, o efeito direto."""
    sc, links, pares, hold = dados
    cen = pd.Series({"XLU": -0.05, "XLE": 0.03, "XLU/0": 0.0} | {f"XLU{k}": -0.06 for k in range(3)})
    op = teia.opcoes(sc, links, pares, hold, None, None, foco="juros", cenario=cen,
                     diretos={("juros", "XLU"): -0.04, ("juros", "XLE"): 0.02})
    nos = {n["id"]: n for n in op["series"][0]["data"]}
    assert "−5,0%" in nos["XLU"]["label"]["formatter"]
    assert nos["XLU/XLU0"]["label"]["show"] and nos["XLU/XLU0"]["label"]["formatter"] == "XLU0 −6,0%"
    assert not nos["XLK/XLK0"]["label"]["show"]                                   # fora da cadeia: sem rótulo
    lig = {(a["source"], a["target"]): a for a in op["series"][0]["links"]}
    assert lig[("juros", "XLU")]["label"]["formatter"] == "−4,0%" and "label" not in lig[("XLU", "XLU/XLU0")]
