"""Configuração comum dos testes.

Os testes não podem depender de rede (CI e reprodutibilidade): qualquer tentativa de abrir
uma ligação falha de imediato. Os dados são sintéticos, gerados em cada teste.
"""
import socket

import pytest


@pytest.fixture(autouse=True)
def _sem_rede(monkeypatch):
    def bloqueado(*args, **kwargs):
        raise RuntimeError("Os testes não podem usar a rede")
    monkeypatch.setattr(socket.socket, "connect", bloqueado)
    monkeypatch.setattr(socket, "create_connection", bloqueado)
