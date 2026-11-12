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


def grande(x: float, d: int = 1) -> str:
    """Valores grandes em português europeu: 4.2e12 → "4,2 biliões"; 8.5e11 → "850,0 mil milhões"."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    for lim, nome in ((1e12, "biliões"), (1e9, "mil milhões"), (1e6, "milhões")):
        if abs(x) >= lim:
            return f"{_fmt(x / lim, d, False)} {nome}"
    return _fmt(x, 0, False)


def data_curta(iso: str | None) -> str:
    """"2026-11-11T23:16:52Z" → "11/11 23:16" (UTC); None → ""."""
    if not iso:
        return ""
    try:
        from datetime import datetime
        t = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        return f"{t.day:02d}/{t.month:02d} {t.hour:02d}:{t.minute:02d}"
    except ValueError:
        return str(iso)[:10]
