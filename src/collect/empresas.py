"""Empresas de cada setor: maiores posições dos ETFs, ficha de bolsa e notícias (Yahoo Finance).

Funções puras (testáveis sem rede): norm_ticker, parse_holdings, fallback_holdings, parse_info, parse_news.
Com rede: fetch_holdings, fetch_info, fetch_news, main (grava data/features/holdings.csv).
"""
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (COMPANY_NEWS, COMPANY_START, EXTRA_TICKERS, FEATURES, HOLDINGS_TOP, RAW, YAHOO_SETOR_ETF,  # noqa: E402
                             SECTOR_PROFILE, SECTORS)

COLS = ["setor", "ticker", "nome", "peso"]
INFO_FIELDS = {  # campo do Yahoo → nome na ficha
    "longName": "nome", "sector": "setor_yahoo", "industry": "industria", "country": "pais", "website": "site",
    "longBusinessSummary": "descricao", "fullTimeEmployees": "empregados", "currency": "moeda",
    "regularMarketPrice": "preco", "previousClose": "fecho_anterior", "regularMarketChangePercent": "var_dia_pct",
    "marketCap": "cap_bolsa", "trailingPE": "per", "forwardPE": "per_futuro", "beta": "beta_yahoo",
    "fiftyTwoWeekLow": "min_52s", "fiftyTwoWeekHigh": "max_52s", "dividendYield": "dividendo_pct",
    "averageVolume": "volume_medio", "recommendationKey": "recomendacao", "targetMeanPrice": "preco_alvo",
}


def norm_ticker(t: str) -> str:
    """Símbolo no formato do Yahoo: BRK.B → BRK-B."""
    return str(t).strip().upper().replace(".", "-")


def parse_holdings(setor: str, raw: pd.DataFrame, top: int = HOLDINGS_TOP) -> pd.DataFrame:
    """Tabela do Yahoo (índice = símbolo; colunas Name, Holding Percent) → setor, ticker, nome, peso."""
    d = raw.reset_index()
    d.columns = ["ticker", "nome", "peso"][: len(d.columns)]
    d = d.assign(setor=setor, ticker=d["ticker"].map(norm_ticker), peso=d["peso"].astype(float))
    return d.sort_values("peso", ascending=False).head(top)[COLS].reset_index(drop=True)


def fallback_holdings(profile: dict = SECTOR_PROFILE) -> pd.DataFrame:
    """Sem dados do Yahoo: as ações de SECTOR_PROFILE com pesos iguais e o ticker como nome."""
    rows = [{"setor": s, "ticker": norm_ticker(a), "nome": a, "peso": 1 / len(p["acoes"])}
            for s, p in profile.items() for a in p["acoes"]]
    return pd.DataFrame(rows, columns=COLS)


def parse_info(raw: dict) -> dict:
    """Só os campos da ficha, com nomes em português; campos em falta ficam de fora."""
    return {pt: raw[k] for k, pt in INFO_FIELDS.items() if raw.get(k) not in (None, "")}


def parse_news(raw: list, n: int = COMPANY_NEWS) -> list[dict]:
    """Notícias do Yahoo (formato novo, com 'content', ou antigo) → titulo, fonte, data, link, resumo."""
    out = []
    for item in raw or []:
        c = item.get("content") or item
        titulo = c.get("title")
        link = ((c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
                or c.get("link"))
        if not titulo or not link:
            continue
        data = c.get("pubDate") or c.get("displayTime")
        if data is None and c.get("providerPublishTime"):
            data = pd.Timestamp(c["providerPublishTime"], unit="s", tz="UTC").isoformat()
        out.append({"titulo": titulo, "link": link, "data": str(data) if data else None,
                    "fonte": (c.get("provider") or {}).get("displayName") or c.get("publisher"),
                    "resumo": c.get("summary") or c.get("description")})
    return out[:n]


# ----------------------------------------------------------------------------- com rede
def fetch_holdings(setor: str, top: int = HOLDINGS_TOP) -> pd.DataFrame:
    return parse_holdings(setor, yf.Ticker(setor).funds_data.top_holdings, top)


def fetch_info(ticker: str) -> dict:
    return parse_info(yf.Ticker(ticker).info)


def fetch_news(ticker: str, n: int = COMPANY_NEWS) -> list[dict]:
    return parse_news(yf.Ticker(ticker).news, n)


def extra_de_info(ticker: str, info: dict, mapa: dict = YAHOO_SETOR_ETF) -> dict | None:
    """Ficha do Yahoo (`parse_info`) → {ticker, nome, setor}: o setor é o ETF da classificação do Yahoo, ou None
    (sem classificação: um ETF, um fundo). None se o Yahoo não conhece o ticker (sem nome)."""
    if not info.get("nome"):
        return None
    return {"ticker": ticker, "nome": info["nome"], "setor": mapa.get(info.get("setor_yahoo"))}


def validar_ticker(texto: str) -> dict | None:
    """Procura o ticker no Yahoo tal como foi escrito (ASML.AS tem o sufixo da bolsa) e, se falhar, no formato
    norte-americano (BRK.B → BRK-B)."""
    for t in dict.fromkeys([str(texto).strip().upper(), norm_ticker(texto)]):
        if t:
            try:
                ficha = extra_de_info(t, fetch_info(t))
            except Exception:  # noqa: BLE001  ticker desconhecido ou Yahoo em baixo
                ficha = None
            if ficha:
                return ficha
    return None


def load_holdings(folder: Path = FEATURES) -> pd.DataFrame:
    """holdings.csv da pasta dada, ou o recurso de SECTOR_PROFILE se não existir."""
    f = folder / "holdings.csv"
    return pd.read_csv(f) if f.exists() else fallback_holdings()


def download_prices(tickers: list[str], start: str = COMPANY_START) -> pd.DataFrame:
    """Fechos ajustados das empresas desde `start`, num único pedido. Empresas sem dados ficam de fora."""
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=False)["Close"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df[[t for t in tickers if t in df and not df[t].dropna().empty]]


def merge_prices(novo: pd.DataFrame, antigo: pd.DataFrame | None, tickers: list[str]) -> pd.DataFrame:
    """Preços novos; as empresas que falharam no download mantêm a série do ficheiro anterior."""
    falta = [t for t in tickers if t not in novo and antigo is not None and t in antigo]
    out = novo if not falta else novo.join(antigo[falta], how="outer")
    return out[[t for t in tickers if t in out]]


def main_prices() -> Path:
    """Histórico de preços das empresas de holdings.csv → data/raw/prices_empresas.parquet."""
    tickers = list(dict.fromkeys(load_holdings()["ticker"]))
    out = RAW / "prices_empresas.parquet"
    antigo = pd.read_parquet(out) if out.exists() else None
    p = merge_prices(download_prices(tickers), antigo, tickers)
    RAW.mkdir(parents=True, exist_ok=True)
    p.to_parquet(out)
    print(f"[coletor] empresas: {p.shape[1]} séries, {p.index.min().date()} → {p.index.max().date()}"
          + (f"; sem dados: {[t for t in tickers if t not in p]}" if p.shape[1] < len(tickers) else ""))
    return out


def main() -> Path:
    """Posições de cada ETF; um setor que falhe mantém as posições do ficheiro anterior (ou o recurso)."""
    antigo = load_holdings()
    partes, falhas = [], []
    for s in SECTORS:
        try:
            partes.append(fetch_holdings(s))
        except Exception:  # noqa: BLE001  Yahoo sem dados do fundo: mantém o que havia
            falhas.append(s)
            partes.append(antigo[(antigo["setor"] == s) & (antigo["peso"] > 0)])     # sem as extra antigas
    FEATURES.mkdir(parents=True, exist_ok=True)
    out = FEATURES / "holdings.csv"
    h = pd.concat(partes, ignore_index=True)
    extra = [x for x in (validar_ticker(t) for t in EXTRA_TICKERS if t not in set(h["ticker"])) if x and x["setor"]]
    if extra:                                          # ações seguidas no pipeline, fora das 10 maiores de cada ETF
        h = pd.concat([h, pd.DataFrame(extra).assign(peso=0.0)[COLS]], ignore_index=True)
    h.to_csv(out, index=False, float_format="%.6f")
    print(f"[coletor] posições: {len(h)} empresas em {h['setor'].nunique()} setores"
          + (f"; mantidas do ficheiro anterior: {falhas}" if falhas else "")
          + (f"; extra: {[x['ticker'] for x in extra]}" if extra else ""))
    return out


if __name__ == "__main__":
    main()
    main_prices()
