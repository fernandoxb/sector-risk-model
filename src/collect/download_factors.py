"""Bot Coletor (fatores macro): níveis diários das séries de MACRO_FACTORS.

FRED via `cockpit.fetch_rates` (fredapi se houver FRED_API_KEY, senão o CSV público);
Yahoo num único pedido, sem threads. Grava data/raw/factors.parquet (uma coluna por nó do mapa).
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import FACTOR_START, MACRO_FACTORS, RAW  # noqa: E402
from src.models.cockpit import fetch_rates  # noqa: E402


def download(start: str = FACTOR_START) -> pd.DataFrame:
    cols = {}
    for k, f in MACRO_FACTORS.items():
        if f["fonte"] == "fred":
            cols[k] = fetch_rates(f["serie"], start)
            if "serie_longa" in f:                       # série com mais história para as janelas longas
                cols[f"{k}_longo"] = fetch_rates(f["serie_longa"], start)
    yahoo = {k: f["serie"] for k, f in MACRO_FACTORS.items() if f["fonte"] == "yahoo"}
    if yahoo:
        df = yf.download(list(yahoo.values()), start=start, auto_adjust=True, progress=False, threads=False)
        close = df["Close"]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        for k, s in yahoo.items():
            cols[k] = close[s]
    out = pd.DataFrame(cols)[[c for c in cols if c in MACRO_FACTORS or c.endswith("_longo")]].sort_index()
    out.index.name = "date"
    missing = [k for k in out if out[k].dropna().empty]
    if missing:
        raise RuntimeError(f"Sem dados para os fatores: {missing}")
    return out


def main() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    f = download()
    out = RAW / "factors.parquet"
    f.to_parquet(out)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (RAW / "factors.meta.txt").write_text(
        f"descarregado_em={stamp}\n" + "".join(f"{k}={f[k].first_valid_index().date()}→"
                                               f"{f[k].last_valid_index().date()}\n" for k in f))
    print(f"[coletor] fatores: " + ", ".join(f"{k} até {f[k].last_valid_index().date()}" for k in f))
    return out


if __name__ == "__main__":
    main()
