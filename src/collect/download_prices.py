"""Bot Coletor: descarrega preços ajustados dos ETFs setoriais + SPY.

Faz um único pedido ao Yahoo (sem threads, para evitar bloqueios e conflitos de cache)
e grava os dados em bruto em data/raw/prices.parquet.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import RAW, START, TICKERS  # noqa: E402


def download(start: str = START) -> pd.DataFrame:
    df = yf.download(TICKERS, start=start, auto_adjust=True, progress=False, threads=False)
    prices = df["Close"].copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "date"
    missing = [t for t in TICKERS if t not in prices or prices[t].dropna().empty]
    if missing:
        raise RuntimeError(f"Sem dados para: {missing}")
    return prices[TICKERS]


def main() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    prices = download()
    out = RAW / "prices.parquet"
    prices.to_parquet(out)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (RAW / "prices.meta.txt").write_text(
        f"descarregado_em={stamp}\nlinhas={len(prices)}\n"
        f"inicio={prices.index.min().date()}\nfim={prices.index.max().date()}\n"
    )
    print(f"[coletor] {len(prices)} dias, {prices.index.min().date()} → {prices.index.max().date()}")
    return out


if __name__ == "__main__":
    main()
