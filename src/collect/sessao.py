"""Última sessão de bolsa de Nova Iorque já fechada (função pura).

Usada na chave de cache do Coletor (dois runs só partilham a coleta se esperarem a mesma sessão)
e no alerta de dados atrasados do Sentinela. Não conhece os feriados dos EUA.
"""
import sys
from datetime import date, time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import MARKET, MARKET_CLOSE, MARKET_TZ  # noqa: E402


def ultima_sessao(agora: pd.Timestamp | None = None, tz: str = MARKET_TZ, fecho: str = MARKET_CLOSE) -> date:
    """Data da última sessão (dia útil) cujo fecho em `tz` já passou de `fecho`.

    `agora` sem fuso horário é lido como UTC. Ex.: quarta 22:30 em Lisboa = 17:30 em NY → quarta;
    quinta 02:00 em Lisboa = quarta 21:00 em NY → quarta; sábado → sexta.
    """
    agora = pd.Timestamp.now(tz="UTC") if agora is None else pd.Timestamp(agora)
    if agora.tzinfo is None:
        agora = agora.tz_localize("UTC")
    t = agora.tz_convert(tz)
    d = t.normalize().tz_localize(None)
    if t.weekday() >= 5 or t.time() < time.fromisoformat(fecho):
        d -= pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d.date()


def ultimo_fecho(prices: pd.DataFrame, col: str = MARKET) -> date:
    """Último dia com preço do mercado (SPY) num DataFrame de preços."""
    return prices[col].dropna().index.max().date()


def dados_completos(ultimo: date, sessao: date) -> bool:
    """Os preços já chegam à sessão esperada?"""
    return ultimo >= sessao
