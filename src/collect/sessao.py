"""Calendário da NYSE e última sessão de Nova Iorque já fechada (funções puras).

Feriados pelas regras da bolsa (sem dependências):
  - data fixa, com compensação: Ano Novo, Juneteenth (desde 2022), Independência e Natal; ao sábado fecha a
    sexta anterior e ao domingo a segunda seguinte. Exceção: o Ano Novo ao sábado não fecha a sexta (31/12).
  - data móvel: Martin Luther King (3.ª seg. jan.), Presidents' Day (3.ª seg. fev.), Sexta-feira Santa
    (Páscoa − 2 dias), Memorial Day (última seg. maio), Labor Day (1.ª seg. set.), Thanksgiving (4.ª qui. nov.).
  - fechos excecionais (furacões, lutos nacionais): lista NYSE_SPECIAL_CLOSURES em config/settings.py.
Validado contra os dias em que o SPY realmente negociou (tests/test_sessao.py).

Fechos antecipados (13:00 em NY), também por regra: 3 de julho, o dia a seguir ao Thanksgiving e a véspera de
Natal, quando são dias de sessão. Não se conseguem validar com preços diários (o dia tem preço na mesma), só
contra o calendário publicado pela bolsa; a regra do 3 de julho teve exceções históricas.

Usado na chave de cache do Coletor, no alerta de dados atrasados, na data da previsão oficial e no dia do
relatório semanal.
"""
import sys
from datetime import date, time, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (MARKET, MARKET_CLOSE, MARKET_EARLY_CLOSE, MARKET_TZ,  # noqa: E402
                             NYSE_JUNETEENTH_FROM, NYSE_SPECIAL_CLOSURES)


# ----------------------------------------------------------------------------- feriados
def pascoa(ano: int) -> date:
    """Domingo de Páscoa (calendário gregoriano, algoritmo anónimo de Meeus/Jones/Butcher)."""
    a, b, c = ano % 19, ano // 100, ano % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = (h + l - 7 * m + 114) % 31 + 1
    return date(ano, mes, dia)


def _n_esimo(ano: int, mes: int, dia_semana: int, n: int) -> date:
    """n-ésimo `dia_semana` (0 = segunda) do mês; n = −1 para o último."""
    if n > 0:
        d = date(ano, mes, 1)
        d += timedelta(days=(dia_semana - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    d = date(ano + (mes == 12), mes % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - dia_semana) % 7)


def _compensado(d: date) -> date | None:
    """Sábado → sexta anterior; domingo → segunda seguinte."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def feriados_nomes(ano: int) -> dict[date, str]:
    """Dias úteis (seg–sex) em que a NYSE está fechada no ano `ano` → nome do feriado."""
    dias = {_compensado(date(ano, 7, 4)): "Dia da Independência", _compensado(date(ano, 12, 25)): "Natal"}
    if ano >= NYSE_JUNETEENTH_FROM:
        dias[_compensado(date(ano, 6, 19))] = "Juneteenth"
    if date(ano, 1, 1).weekday() != 5:              # Ano Novo ao sábado: não há compensação na sexta
        dias[_compensado(date(ano, 1, 1))] = "Ano Novo"
    dias |= {_n_esimo(ano, 1, 0, 3): "Martin Luther King", _n_esimo(ano, 2, 0, 3): "Presidents' Day",
             pascoa(ano) - timedelta(days=2): "Sexta-feira Santa", _n_esimo(ano, 5, 0, -1): "Memorial Day",
             _n_esimo(ano, 9, 0, 1): "Labor Day", _n_esimo(ano, 11, 3, 4): "Thanksgiving"}
    dias |= {date.fromisoformat(d): nome for d, nome in NYSE_SPECIAL_CLOSURES.items() if d.startswith(str(ano))}
    return {d: n for d, n in sorted(dias.items()) if d.year == ano and d.weekday() < 5}


def feriados_nyse(ano: int) -> frozenset[date]:
    return frozenset(feriados_nomes(ano))


def dia_util(d: date) -> bool:
    """Há sessão na NYSE neste dia?"""
    return d.weekday() < 5 and d not in feriados_nyse(d.year)


def proximo_dia_util(d: date) -> date:
    d += timedelta(days=1)
    while not dia_util(d):
        d += timedelta(days=1)
    return d


def dia_util_anterior(d: date) -> date:
    d -= timedelta(days=1)
    while not dia_util(d):
        d -= timedelta(days=1)
    return d


def fecho_antecipado(d: date) -> bool:
    """A NYSE fecha às 13:00 neste dia? (3 de julho, dia a seguir ao Thanksgiving, véspera de Natal)"""
    if not dia_util(d):
        return False
    return ((d.month, d.day) in ((7, 3), (12, 24))
            or d == _n_esimo(d.year, 11, 3, 4) + timedelta(days=1))


def proximos_feriados(d: date, n: int = 3) -> list[tuple[date, str]]:
    """Os `n` feriados da NYSE a partir de `d` (inclusive)."""
    todos = [(f, nome) for ano in (d.year, d.year + 1) for f, nome in feriados_nomes(ano).items() if f >= d]
    return todos[:n]


def proximo_fecho_antecipado(d: date) -> date:
    """O próximo dia com fecho às 13:00, a partir de `d` (inclusive)."""
    while not fecho_antecipado(d):
        d += timedelta(days=1)
    return d


def dias_uteis(inicio: date, fim: date) -> list[date]:
    """Sessões da NYSE entre `inicio` e `fim` (inclusive)."""
    return [d.date() for d in pd.date_range(inicio, fim, freq="D") if dia_util(d.date())]


# ----------------------------------------------------------------------------- sessão esperada
def ultima_sessao(agora: pd.Timestamp | None = None, tz: str = MARKET_TZ, fecho: str = MARKET_CLOSE) -> date:
    """Data da última sessão da NYSE cujo fecho em `tz` já passou de `fecho`.

    `agora` sem fuso horário é lido como UTC. Ex.: quarta 22:30 em Lisboa = 17:30 em NY → quarta;
    quinta 02:00 em Lisboa = quarta 21:00 em NY → quarta; sábado → sexta; dia de Thanksgiving → quarta.
    Nos dias de fecho antecipado (13:00) a sessão conta como fechada às MARKET_EARLY_CLOSE.
    """
    agora = pd.Timestamp.now(tz="UTC") if agora is None else pd.Timestamp(agora)
    if agora.tzinfo is None:
        agora = agora.tz_localize("UTC")
    t = agora.tz_convert(tz)
    d = t.date()
    if dia_util(d) and t.time() >= time.fromisoformat(MARKET_EARLY_CLOSE if fecho_antecipado(d) else fecho):
        return d
    return dia_util_anterior(d)


def ultimo_fecho(prices: pd.DataFrame, col: str = MARKET) -> date:
    """Último dia com preço do mercado (SPY) num DataFrame de preços."""
    return prices[col].dropna().index.max().date()


def dados_completos(ultimo: date, sessao: date) -> bool:
    """Os preços já chegam à sessão esperada?"""
    return ultimo >= sessao
