"""Testes da formatação em português (src/report/formato.py)."""
from src.report import formato as fm


def test_percentagens_e_sinais():
    assert fm.pct(0.0908) == "9,1%" and fm.pct(0.09078, 2) == "9,08%"
    assert fm.pct(-0.495) == "−49,5%" and fm.pct(0.45, 0, sinal=True) == "+45%"
    assert fm.pct(-0.0001) == "0,0%"                     # sem "−0,0%"
    assert fm.pct(float("nan")) == "—"


def test_numeros_pp_multiplos_e_nivel():
    assert fm.num(1234.5) == "1 234,50" and fm.num(-1.36, 2) == "−1,36"
    assert fm.pp(0.16) == "+0,16 p.p." and fm.pp(-0.3) == "−0,30 p.p."
    assert fm.mult(1.4912) == "1,49×"
    assert fm.nivel(0.975) == "97,5%" and fm.nivel(0.99) == "99%" and fm.nivel(0.95) == "95%"
