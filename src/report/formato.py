"""Formatação de números para a interface, em português (vírgula decimal, sinal "−" tipográfico).

Funções puras, usadas pelo cockpit, para que os números saiam iguais em todo o lado.
"""
import math

MENOS = "−"   # U+2212


def _fmt(x: float, d: int, sinal: bool) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if round(x, d) == 0:
        x = 0.0                                # evita "−0,0"
    s = f"{abs(x):,.{d}f}".replace(",", " ").replace(".", ",")
    return (MENOS if x < 0 else "+" if sinal and x > 0 else "") + s


def num(x: float, d: int = 2, sinal: bool = False) -> str:
    """1234.5 → "1 234,50"; com sinal: "+0,25"."""
    return _fmt(x, d, sinal)


def pct(x: float, d: int = 1, sinal: bool = False) -> str:
    """0.0908 → "9,1%"; -0.495 → "−49,5%"."""
    out = _fmt(None if x is None else x * 100, d, sinal)
    return out if out == "—" else out + "%"


def pp(x: float, d: int = 2) -> str:
    """Variação em pontos percentuais, sempre com sinal: 0.16 → "+0,16 p.p."."""
    return _fmt(x, d, True) + " p.p."


def mult(x: float, d: int = 2) -> str:
    """1.49 → "1,49×"."""
    return _fmt(x, d, False) + "×"


def nivel(alpha: float) -> str:
    """0.975 → "97,5%"; 0.99 → "99%" (nunca arredonda o nível)."""
    s = f"{alpha * 100:.1f}".rstrip("0").rstrip(".")
    return s.replace(".", ",") + "%"
