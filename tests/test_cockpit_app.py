"""Teste de fumo do cockpit: corre app/cockpit/app.py inteiro em modo snapshot (sem rede) e verifica
que não há exceções e que os 8 separadores aparecem. Usa o AppTest do Streamlit."""
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

from pathlib import Path  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "cockpit" / "app.py"


def test_cockpit_corre_em_modo_snapshot(monkeypatch):
    monkeypatch.setenv("COCKPIT_SNAPSHOT", "1")
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs][:7] == ["📋 Resumo", "🕸️ Teia de risco", "💼 Minha carteira", "📊 Análise ao vivo",
                                              "🧪 Modelo", "🧩 Carteira-base", "📖 Glossário"]
    texto = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "Modo snapshot" in " ".join(i.value for i in at.info)
    assert "97,5%" in texto and "EVT" in texto


def test_mapa_de_risco_abre_setor_e_empresa(monkeypatch):
    """Setor aberto → tabela das empresas; empresa escolhida → ficha com dados do snapshot (sem rede)."""
    monkeypatch.setenv("COCKPIT_SNAPSHOT", "1")
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.session_state["mapa_setor"] = "XLK"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Empresas do Tecnologia (XLK)" in s.value for s in at.subheader)
    assert "Score de risco" in [m.label for m in at.metric]            # ficha do setor, agora dentro do mapa
    at.session_state["mapa_empresa"] = "NVDA"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("(NVDA)" in s.value for s in at.subheader)
    assert any("Últimas notícias" in m.value for m in at.markdown)


def test_minha_carteira_do_link(monkeypatch):
    """Carteira lida do link (?carteira=...): métricas, mapa e stress sem exceções, em modo snapshot."""
    monkeypatch.setenv("COCKPIT_SNAPSHOT", "1")
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.query_params["carteira"] = "NVDA:40,JPM:30,XLU:20,SPY:10"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    rotulos = [m.label for m in at.metric]
    assert "Quem mais pesa no risco" in rotulos and any(r.startswith("Carteira com o SPY") for r in rotulos)
    quem = next(m for m in at.metric if m.label == "Quem mais pesa no risco")
    assert quem.value == "NVDA"


def test_previsao_fhs_da_carteira_e_modelo_na_ficha(monkeypatch):
    """A carteira tem ES previsto pelo modelo oficial e a ficha da empresa mostra o estado do backtest."""
    monkeypatch.setenv("COCKPIT_SNAPSHOT", "1")
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.query_params["carteira"] = "NVDA:40,JPM:30,XLU:20,SPY:10"
    at.query_params["setor"] = "XLK"
    at.query_params["empresa"] = "NVDA"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    rotulos = [m.label for m in at.metric]
    assert sum(r.startswith("ES 97,5% previsto · ") for r in rotulos) >= 2      # carteira e ficha da NVDA
    assert "Backtest do modelo" in rotulos and "VaR 99% previsto" in rotulos
    assert "Backtest desta carteira" in rotulos


def test_empresa_sem_backtest_nao_aparece_validada():
    import sys
    sys.path.insert(0, str(APP.parent))
    from paineis.comum import modelo_txt
    assert modelo_txt({"validado": float("nan"), "es_subestimado": float("nan"), "mult_es": 1.0}) == "sem backtest"
    assert modelo_txt({"validado": True, "es_subestimado": False, "mult_es": 1.0}) == "✅ validado"


@pytest.mark.parametrize("janela, anos", [("1 Ano", 1), ("10 Anos", 10)])
def test_janela_muda_as_ligacoes_do_mapa(monkeypatch, janela, anos):
    """A janela da barra lateral escolhe as quedas conjuntas dessa janela (q de 10% com 1 ano) e os preços de 10 anos."""
    monkeypatch.setenv("COCKPIT_SNAPSHOT", "1")
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.sidebar.selectbox[0].set_value(janela).run()
    assert not at.exception, [e.value for e in at.exception]
    texto = " ".join(c.value for c in at.caption)
    assert ("10% piores dias" in texto) == (anos == 1)                  # com 1 ano a cauda é de 10%
    at.session_state["mapa_setor"] = "XLK"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Empresas do Tecnologia (XLK)" in s.value for s in at.subheader)


def test_cenario_na_teia(monkeypatch):
    """Com um choque nos juros, o painel mostra a tabela dos setores e a teia corre sem erros."""
    monkeypatch.setenv("COCKPIT_SNAPSHOT", "1")
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.session_state["cen_fatores"] = ["juros"]
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("O que o modelo espera para os outros fatores" in m.value for m in at.markdown)
    assert any("Os intervalos são do modelo" in w.value for w in at.warning)          # cobertura medida, declarada


def test_teia_mostra_score_fatores_e_a_carteira_no_cenario(monkeypatch):
    """Sem setor aberto, por baixo da teia aparece o score dos setores (ou os fatores macro); com um choque, a carteira
    da Minha carteira tem o seu resultado no cenário."""
    monkeypatch.setenv("COCKPIT_SNAPSHOT", "1")
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Scorecard de risco por setor" in s.value for s in at.subheader)
    at.session_state["teia_vista"] = "fatores"
    at.session_state["cen_fatores"] = ["juros"]
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Como os fatores macro mexem com cada setor" in s.value for s in at.subheader)
    assert any("A tua carteira" in m.value for m in at.markdown)       # a carteira de exemplo da Minha carteira
