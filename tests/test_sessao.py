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
