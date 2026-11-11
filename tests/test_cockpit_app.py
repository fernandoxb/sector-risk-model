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
    assert [t.label for t in at.tabs][:8] == ["📋 Resumo", "🎯 Setores", "🔎 Ficha do setor", "📊 Análise ao vivo", "🔗 Fatores macro",
                                              "🧪 Modelo (backtest)", "🧩 Carteira e stress", "📖 Glossário"]
    texto = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "Modo snapshot" in " ".join(i.value for i in at.info)
    assert "97,5%" in texto and "FHS" in texto
