"""O flow pára no primeiro bot que falha: dados rejeitados pelo Inspetor nunca chegam aos modelos nem ao site."""
import logging

import pytest

pytest.importorskip("prefect")          # o CI instala só requirements-test.txt, sem o Prefect
from flows import pipeline as fl  # noqa: E402

DEPOIS = ("modelador", "backtester", "backtester_empresas", "carteira", "score", "ligacoes", "dependencia_cauda",
          "sentinela", "relatorio", "snapshot_cockpit", "publicador")


def test_inspetor_que_falha_para_o_flow(monkeypatch):
    correram = []

    def falha():
        raise ValueError("dados inválidos")

    monkeypatch.setattr(fl, "get_run_logger", lambda: logging.getLogger("teste"))
    monkeypatch.setattr(fl, "inspetor", falha)
    for nome in DEPOIS:
        monkeypatch.setattr(fl, nome, lambda *a, _n=nome, **k: correram.append(_n))
    with pytest.raises(ValueError):
        fl.pipeline.fn(offline=True)
    assert correram == []
