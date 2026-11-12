"""Testes da última sessão de NY fechada (src/collect/sessao.py) e da junção das ações do snapshot."""
from datetime import date

import pandas as pd

from src.collect.sessao import ultima_sessao
from src.report import snapshot as sn


def test_ultima_sessao():
    lis = "Europe/Lisbon"
    assert ultima_sessao(pd.Timestamp("2026-11-11 22:30", tz=lis)) == date(2026, 11, 11)   # 17:30 em NY
    assert ultima_sessao(pd.Timestamp("2026-11-11 02:04", tz=lis)) == date(2026, 11, 10)   # véspera em NY
    assert ultima_sessao(pd.Timestamp("2026-11-11 21:50", tz=lis)) == date(2026, 11, 10)   # 16:50 em NY: ainda não
    assert ultima_sessao(pd.Timestamp("2026-11-11 22:00", tz=lis)) == date(2026, 11, 11)   # 17:00 em NY
    assert ultima_sessao(pd.Timestamp("2026-11-14 12:00", tz=lis)) == date(2026, 11, 13)   # sábado → sexta
    assert ultima_sessao(pd.Timestamp("2026-11-16 10:00", tz=lis)) == date(2026, 11, 13)   # segunda de manhã
    assert ultima_sessao(pd.Timestamp("2026-11-11 22:30")) == date(2026, 11, 11)          # sem fuso = UTC


def test_acoes_em_falta_vem_do_snapshot_anterior():
    idx_old = pd.bdate_range("2026-11-02", periods=6)
    idx_new = pd.bdate_range("2026-11-02", periods=7)
    old = pd.DataFrame({"A": 1.0, "B": 2.0, "C": 3.0}, index=idx_old)
    new = pd.DataFrame({"A": 10.0, "C": 30.0}, index=idx_new)          # B falhou no download
    m, falta = sn.merge_stocks(new, old, ["A", "B", "C"])
    assert falta == ["B"] and list(m.columns) == ["A", "B", "C"]
    assert m["A"].iloc[-1] == 10.0 and m["B"].iloc[0] == 2.0 and pd.isna(m["B"].iloc[-1])


def test_ultimo_fecho_e_dados_completos():
    from src.collect.sessao import dados_completos, ultimo_fecho
    p = pd.DataFrame({"SPY": [1.0, 2.0, None], "XLK": [1.0, 2.0, 3.0]},
                     index=pd.to_datetime(["2026-11-09", "2026-11-10", "2026-11-11"]))
    assert ultimo_fecho(p) == date(2026, 11, 10)                  # linha incompleta no fim não conta
    assert dados_completos(date(2026, 11, 11), date(2026, 11, 11))
    assert not dados_completos(date(2026, 11, 10), date(2026, 11, 11))


# ----------------------------------------------------------------------------- calendário da NYSE
from src.collect import sessao as ss  # noqa: E402
from src.report import weekly as wk  # noqa: E402


def test_pascoa():
    assert [ss.pascoa(a) for a in (2008, 2024, 2025, 2026)] == [date(2008, 3, 23), date(2024, 3, 31),
                                                                 date(2025, 4, 20), date(2026, 4, 5)]


def test_feriados_conhecidos():
    f = lambda a: sorted(str(d) for d in ss.feriados_nyse(a))  # noqa: E731
    assert f(2022) == ["2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30", "2022-06-20", "2022-07-04",
                       "2022-09-05", "2022-11-24", "2022-12-26"]              # Ano Novo ao sábado: sem compensação
    assert f(2026) == ["2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
                       "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"]  # 4 de julho ao sábado → sexta
    assert date(2021, 6, 18) not in ss.feriados_nyse(2021)                     # Juneteenth só desde 2022
    assert date(2025, 1, 9) in ss.feriados_nyse(2025)                          # fecho excecional (luto)


def test_calendario_bate_com_as_sessoes_reais_do_spy():
    """Validação independente: os dias em que o SPY negociou (snapshot versionado, 3 anos) são exatamente
    as sessões do calendário."""
    p = pd.read_parquet(sn.SNAPSHOT / "prices.parquet")
    real = set(p["SPY"].dropna().index.date)
    assert real == set(ss.dias_uteis(min(real), max(real)))


def test_sessao_e_relatorio_com_feriados():
    lis = "Europe/Lisbon"
    assert ultima_sessao(pd.Timestamp("2026-11-26 23:00", tz=lis)) == date(2026, 11, 25)   # Thanksgiving
    assert ss.proximo_dia_util(date(2026, 11, 25)) == date(2026, 11, 27)
    assert ss.proximo_dia_util(date(2026, 4, 2)) == date(2026, 4, 6)                        # Sexta-feira Santa
    assert wk.is_report_day("2026-04-02") and not wk.is_report_day("2026-04-01")            # quinta fecha a semana
    assert wk.is_report_day("2026-11-13") and not wk.is_report_day("2026-11-12")


def test_fechos_antecipados_e_proximas_datas():
    lis = "Europe/Lisbon"
    assert ss.fecho_antecipado(date(2026, 11, 27)) and ss.fecho_antecipado(date(2026, 12, 24))
    assert ss.fecho_antecipado(date(2025, 7, 3)) and not ss.fecho_antecipado(date(2026, 7, 3))   # 2026: feriado
    assert not ss.fecho_antecipado(date(2027, 12, 24))                 # Natal ao sábado: dia 24 é feriado
    assert not ss.fecho_antecipado(date(2026, 11, 25))
    # sexta 27/11/2026: 14:30 em NY (19:30 em Lisboa) já conta como fechada; num dia normal ainda não
    assert ultima_sessao(pd.Timestamp("2026-11-27 19:30", tz=lis)) == date(2026, 11, 27)
    assert ultima_sessao(pd.Timestamp("2026-11-27 18:30", tz=lis)) == date(2026, 11, 25)
    assert ultima_sessao(pd.Timestamp("2026-11-20 19:30", tz=lis)) == date(2026, 11, 19)
    assert ss.proximos_feriados(date(2026, 11, 12), 3) == [(date(2026, 11, 26), "Thanksgiving"),
                                                           (date(2026, 12, 25), "Natal"),
                                                           (date(2027, 1, 1), "Ano Novo")]
    assert ss.proximo_fecho_antecipado(date(2026, 11, 12)) == date(2026, 11, 27)
    assert ss.feriados_nomes(2025)[date(2025, 1, 9)].startswith("luto")
