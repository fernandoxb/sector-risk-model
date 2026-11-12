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
    assert [t.label for t in at.tabs][:9] == ["📋 Resumo", "🕸️ Mapa de risco", "🎯 Setores", "🔎 Ficha do setor", "📊 Análise ao vivo", "🔗 Fatores macro",
                                              "🧪 Modelo", "🧩 Carteira", "📖 Glossário"]
    texto = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "Modo snapshot" in " ".join(i.value for i in at.info)
    assert "97,5%" in texto and "FHS" in texto


def test_mapa_de_risco_abre_setor_e_empresa(monkeypatch):
    """Setor aberto → tabela das empresas; empresa escolhida → ficha com dados do snapshot (sem rede)."""
    monkeypatch.setenv("COCKPIT_SNAPSHOT", "1")
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.session_state["mapa_setor"] = "XLK"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Empresas do Tecnologia (XLK)" in s.value for s in at.subheader)
    at.session_state["mapa_empresa"] = "NVDA"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("(NVDA)" in s.value for s in at.subheader)
    assert any("Últimas notícias" in m.value for m in at.markdown)
