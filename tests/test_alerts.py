"""Testes do bot Sentinela e do relatório semanal (src/report/alerts.py, src/report/weekly.py).

Estados sintéticos de ontem e de hoje; nada toca no cofre Obsidian nem em data/.
"""
import copy

import pandas as pd
import pytest

from src.report import alerts as al
from src.report import weekly as wk


def setor(score=50.0, rank=1, semaforo="verde", p=3, i=3, perda=None, var=0.02, es=0.03):
    s = {"score": score, "rank": rank, "semaforo": semaforo, "impacto": i, "probabilidade": p,
         "pi": p * i, "nivel": al.pi_level(p, i)}
    if perda is not None:
        s |= {"perda": perda, "var975": var, "es975": es, "excecao": perda > var}
    return s


@pytest.fixture
def ontem():
    return {"asof": "2026-11-09",
            "setores": {"XLK": setor(80, 1, p=4, i=4), "XLE": setor(60, 2), "XLU": setor(20, 3, p=1, i=2)},
            "ligacoes": ["juros|XLRE|+", "vix|XLK|+"], "regime_juros": "estável"}


def tipos(alerts):
    return {(a["tipo"], a["alvo"], a["severidade"]) for a in alerts}


# ----------------------------------------------------------------------------- níveis P × I
@pytest.mark.parametrize("p,i,nivel", [(1, 1, "baixo"), (2, 2, "baixo"), (1, 5, "moderado"), (3, 3, "moderado"),
                                       (2, 5, "elevado"), (4, 4, "elevado"), (4, 5, "crítico"), (5, 5, "crítico")])
def test_niveis_pi(p, i, nivel):
    assert al.pi_level(p, i) == nivel


# ----------------------------------------------------------------------------- sem mudanças
def test_sem_mudancas_sem_alertas(ontem):
    hoje = copy.deepcopy(ontem) | {"asof": "2026-11-10"}
    assert al.compare(ontem, hoje) == []


# ----------------------------------------------------------------------------- semáforo
def test_semaforo_severidades(ontem):
    hoje = copy.deepcopy(ontem) | {"asof": "2026-11-10"}
    hoje["setores"]["XLE"]["semaforo"] = "amarelo"
    hoje["setores"]["XLK"]["semaforo"] = "vermelho"
    assert tipos(al.compare(ontem, hoje)) == {("semáforo", "XLE", "média"), ("semáforo", "XLK", "alta")}
    # melhoria = baixa
    assert tipos(al.compare(hoje, ontem)) == {("semáforo", "XLE", "baixa"), ("semáforo", "XLK", "baixa")}


# ----------------------------------------------------------------------------- exceção ao VaR oficial
def test_excecao_media_e_alta(ontem):
    hoje = copy.deepcopy(ontem) | {"asof": "2026-11-10"}
    hoje["setores"]["XLE"] = setor(60, 2, perda=0.025, var=0.02, es=0.03)     # acima do VaR, abaixo do ES
    hoje["setores"]["XLU"] = setor(20, 3, p=1, i=2, perda=0.04, var=0.02, es=0.03)   # acima do ES
    hoje["setores"]["XLK"] = setor(80, 1, p=4, i=4, perda=0.019, var=0.02)       # abaixo do VaR: nada
    a = al.compare(ontem, hoje)
    assert tipos(a) == {("exceção", "XLE", "média"), ("exceção", "XLU", "alta")}
    assert a[0]["severidade"] == "alta"                                      # ordenado por severidade
    assert "97,5%" in a[0]["texto"] and "4,0%" in a[0]["texto"]
    # exceções saem mesmo no primeiro run (sem estado anterior); o resto não
    assert tipos(al.compare(None, hoje)) == {("exceção", "XLE", "média"), ("exceção", "XLU", "alta")}


# ----------------------------------------------------------------------------- nível P × I
def test_nivel_pi(ontem):
    hoje = copy.deepcopy(ontem) | {"asof": "2026-11-10"}
    hoje["setores"]["XLK"] |= {"probabilidade": 5, "pi": 20, "nivel": al.pi_level(5, 4)}   # elevado → crítico
    hoje["setores"]["XLE"] |= {"probabilidade": 4, "pi": 12, "nivel": al.pi_level(4, 3)}   # moderado → elevado
    hoje["setores"]["XLU"] |= {"probabilidade": 2, "pi": 4, "nivel": al.pi_level(2, 2)}    # baixo → baixo: nada
    assert tipos(al.compare(ontem, hoje)) == {("nível P×I", "XLK", "alta"), ("nível P×I", "XLE", "média")}
    assert tipos(al.compare(hoje, ontem)) == {("nível P×I", "XLK", "baixa"), ("nível P×I", "XLE", "baixa")}


# ----------------------------------------------------------------------------- ligações e regime
def test_ligacoes_entram_e_saem(ontem):
    hoje = copy.deepcopy(ontem) | {"asof": "2026-11-10", "ligacoes": ["juros|XLRE|+", "petroleo|XLE|-"]}
    a = al.compare(ontem, hoje)
    assert tipos(a) == {("ligação", "petroleo→XLE", "média"), ("ligação", "vix→XLK", "baixa")}
    assert any("atenua" in x["texto"] for x in a) and any("deixou" in x["texto"] for x in a)


def test_regime_de_juros(ontem):
    hoje = copy.deepcopy(ontem) | {"asof": "2026-11-10", "regime_juros": "subida"}
    assert tipos(al.compare(ontem, hoje)) == {("regime de juros", "juros", "alta")}
    hoje["regime_juros"] = "descida"
    assert tipos(al.compare(ontem, hoje)) == {("regime de juros", "juros", "média")}


# ----------------------------------------------------------------------------- estado a partir dos ficheiros
def test_build_state():
    sc = pd.DataFrame({"score_final": [70.0, 30.0], "rank": [1, 2], "semaforo": ["verde", "amarelo"],
                       "impacto": [5, 1], "probabilidade": [4, 2]}, index=["XLK", "XLU"])
    series = pd.DataFrame({"data": pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-09"]),
                           "setor": ["XLK", "XLK", "XLU"], "ret": [-0.10, -0.03, 0.01],
                           "var_fhs_0.975": [0.02, 0.02, 0.02], "es_fhs_0.975": [0.03, 0.025, 0.03]})
    links = pd.DataFrame({"fator": ["vix", "juros"], "setor": ["XLK", "XLU"], "aceite": [True, False],
                          "sinal": ["+", None]})
    s = al.build_state("2026-11-09", sc, series, links, {"regime": "estável"})
    assert s["setores"]["XLK"]["excecao"] and s["setores"]["XLK"]["perda"] == pytest.approx(0.03)
    assert not s["setores"]["XLU"]["excecao"]                    # ganho de 1%
    assert s["setores"]["XLK"]["nivel"] == "crítico" and s["setores"]["XLU"]["pi"] == 2
    assert s["ligacoes"] == ["vix|XLK|+"] and s["regime_juros"] == "estável"


# ----------------------------------------------------------------------------- Alertas.md e notificação
def test_alertas_md_mais_recente_no_topo_e_idempotente(ontem):
    a1 = [{"severidade": "média", "tipo": "semáforo", "alvo": "XLE", "texto": "[[XLE]]: semáforo verde → amarelo"}]
    t = al.update_alerts(None, "2026-11-09", [], first_run=True)
    t = al.update_alerts(t, "2026-11-10", a1)
    assert t.index("## 2026-11-10") < t.index("## 2026-11-09")
    assert "[severidade:: média]" in t and "Primeiro estado" in t
    assert al.update_alerts(t, "2026-11-10", a1) == t           # re-run do mesmo dia não duplica
    assert "ultimo_dia: 2026-11-10" in t and "alertas_ultimo_dia: 1" in t
    assert al.update_alerts(t, "2026-11-11", [], keep=2).count("\n## ") == 2


def test_notificacao_so_para_alta():
    baixa = [{"severidade": "média", "texto": "x"}, {"severidade": "baixa", "texto": "y"}]
    assert al.notification(baixa) is None
    alta = [{"severidade": "alta", "texto": f"[[XL{i}]]: z"} for i in range(5)]
    msg = al.notification(alta)
    assert "[[" not in msg and msg.endswith("(+2)")


def test_estado_anterior(tmp_path):
    for d in ("2026-11-05", "2026-11-06", "2026-11-09"):
        (tmp_path / f"estado_{d}.json").write_text(f'{{"asof": "{d}"}}')
    assert al.previous_state("2026-11-09", tmp_path)["asof"] == "2026-11-06"
    assert al.previous_state("2026-11-09", tmp_path) == al.previous_state("2026-11-07", tmp_path)
    assert al.previous_state("2026-11-05", tmp_path) is None


# ----------------------------------------------------------------------------- relatório semanal
def test_semana_e_excecoes_da_semana():
    assert wk.week_id("2026-11-13") == "2026-W46" and wk.week_start("2026-11-13") == pd.Timestamp("2026-11-09")
    assert wk.is_report_day("2026-11-13") and not wk.is_report_day("2026-11-12")
    series = pd.DataFrame({"data": pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-13"]),
                           "setor": "XLK", "ret": [-0.05, -0.03, -0.01, -0.05],
                           "var_fhs_0.975": 0.02, "es_fhs_0.975": 0.04})
    exc = wk.week_exceptions(series, "2026-11-12")         # sexta anterior e dia 13 ficam de fora
    assert exc["data"].astype(str).tolist() == ["2026-11-09"] and not exc["acima_es"].any()
    assert wk.week_days(series, "2026-11-13") == 3


def test_stress_mais_severo():
    st = pd.DataFrame({"cenario": ["A"] * 3 + ["B"] * 3, "serie": ["Carteira", "XLK", "XLF"] * 2,
                       "perda": [0.2, 0.3, 0.1, 0.4, 0.2, 0.6], "proxy": [False, False, False, True, False, True]})
    w = wk.worst_stress(st)
    assert w["cenario"] == "B" and w["setor"] == "XLF" and w["proxy_setor"] and w["perda_carteira"] == 0.4


def test_nota_semanal(ontem):
    hoje = copy.deepcopy(ontem) | {"asof": "2026-11-13", "ligacoes": ["juros|XLRE|+", "petroleo|XLE|-"]}
    hoje["setores"]["XLE"] |= {"score": 85.0, "rank": 1}
    hoje["setores"]["XLK"] |= {"rank": 2}
    exc = pd.DataFrame(columns=["data", "setor", "perda", "var", "es", "acima_es"])
    st = {"cenario": "2008 · Crise", "perda_carteira": 0.5, "proxy": True, "setor": "XLF",
          "perda_setor": 0.8, "proxy_setor": False}
    fm, md = wk.weekly_note(hoje, ontem, exc, st, dias=5)
    assert fm["semana"] == "2026-W46" and fm["top1"] == "XLE" and fm["top3"] == "[XLE, XLK, XLU]"
    assert fm["ligacoes_novas"] == 1 and fm["ligacoes_sairam"] == 1 and fm["excecoes"] == 0
    assert "+25,0 (+1 pos.)" in md and "Nenhuma exceção" in md and "1,4 (2,5% × 5 dias" in md
    assert md.startswith("---\ntipo: relatorio-semanal")
    _, md0 = wk.weekly_note(hoje, None, exc, st, dias=5)
    assert "Sem estado anterior" in md0


def test_estado_de_referencia_semanal(tmp_path):
    (tmp_path / "estado_2026-11-10.json").write_text('{"asof": "2026-11-10"}')
    (tmp_path / "estado_2026-11-11.json").write_text('{"asof": "2026-11-11"}')
    assert wk.reference_state("2026-11-13", tmp_path)["asof"] == "2026-11-10"   # 1.ª semana: início da semana
    (tmp_path / "estado_2026-11-06.json").write_text('{"asof": "2026-11-06"}')
    assert wk.reference_state("2026-11-13", tmp_path)["asof"] == "2026-11-06"   # sexta anterior
    assert wk.reference_state("2026-11-10", tmp_path)["asof"] == "2026-11-06"



def test_limite_minimo_de_cada_nivel():
    assert [al.pi_min(n) for n in ("baixo", "moderado", "elevado", "crítico")] == [1, 5, 10, 17]
    assert al.pi_level(4, 4) == "elevado" and al.pi_level(17, 1) == "crítico"


def test_historico_do_score(ontem):
    hoje = copy.deepcopy(ontem) | {"asof": "2026-11-10"}
    hoje["setores"]["XLK"]["score"] = 85.0
    h = al.score_history([ontem, hoje])
    assert len(h) == 6 and list(h.columns) == ["data", "setor", "score", "rank", "pi", "nivel", "semaforo"]
    assert h.query("setor == 'XLK'")["score"].tolist() == [80, 85.0]


def test_alerta_de_dados_atrasados():
    assert al.stale_alert("2026-11-11", "2026-11-11") is None
    a = al.stale_alert("2026-11-10", "2026-11-11")
    assert a["severidade"] == "alta" and "2026-11-10" in a["texto"] and "2026-11-11" in a["texto"]
