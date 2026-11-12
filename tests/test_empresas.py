"""Testes da coleta de empresas (src/collect/empresas.py) e da ficha no snapshot: só funções puras, sem rede."""
import pandas as pd

from src.collect import empresas as emp
from src.report import snapshot as sn


def test_posicoes_do_etf():
    raw = pd.DataFrame({"Name": ["Berkshire B", "JPMorgan", "Visa"], "Holding Percent": [0.12, 0.11, 0.13]},
                       index=pd.Index(["BRK.B", "JPM", "V"], name="Symbol"))
    h = emp.parse_holdings("XLF", raw, top=2)
    assert h["ticker"].tolist() == ["V", "BRK-B"] and h["setor"].unique().tolist() == ["XLF"]
    assert list(h.columns) == emp.COLS
    fb = emp.fallback_holdings({"XLK": {"acoes": ["AAPL", "MSFT"]}})
    assert fb["peso"].tolist() == [0.5, 0.5] and fb["ticker"].tolist() == ["AAPL", "MSFT"]


def test_ficha_e_noticias():
    info = emp.parse_info({"longName": "NVIDIA", "marketCap": 4e12, "trailingPE": None, "lixo": 1})
    assert info == {"nome": "NVIDIA", "cap_bolsa": 4e12}
    novo = {"content": {"title": "T1", "pubDate": "2026-11-11T23:16:52Z", "provider": {"displayName": "Reuters"},
                        "canonicalUrl": {"url": "https://exemplo.invalid/1"}, "summary": "s"}}
    antigo = {"title": "T2", "link": "https://exemplo.invalid/2", "publisher": "AP", "providerPublishTime": 1794000000}
    sem_link = {"content": {"title": "T3"}}
    n = emp.parse_news([novo, antigo, sem_link], n=5)
    assert [x["titulo"] for x in n] == ["T1", "T2"] and n[0]["fonte"] == "Reuters" and n[1]["fonte"] == "AP"
    assert n[1]["data"].startswith("2026-11") and len(emp.parse_news([novo] * 9, n=3)) == 3


def test_ficha_que_falha_mantem_a_anterior():
    def info(t):
        if t == "B":
            raise RuntimeError("Yahoo em baixo")
        return {"nome": t}
    out, falhas = sn.companies_payload(["A", "B", "C"], {"B": {"info": {"nome": "antigo"}, "noticias": []}},
                                       info, lambda t: [])
    assert falhas == ["B"] and out["B"]["info"]["nome"] == "antigo" and out["A"]["info"] == {"nome": "A"}
    assert list(out) == ["A", "B", "C"]
