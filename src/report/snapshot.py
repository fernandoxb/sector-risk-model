"""Snapshot do cockpit (app/cockpit/snapshot/): ficheiros pequenos e versionados para correr sem rede.

O cockpit tenta sempre os dados ao vivo (Yahoo e FRED). Se falharem, ou com COCKPIT_SNAPSHOT=1
(Streamlit Community Cloud), lê daqui. Conteúdo:
  prices.parquet    fechos ajustados dos 11 setores, SPY e ações do microscópio (últimos SNAPSHOT_YEARS anos)
  rates.parquet     Treasury 10Y (DGS10) desde SNAPSHOT_RATE_MARGIN_DAYS dias antes do 1.º preço
  <features>        cópias de data/features usadas pelo cockpit (séries do backtest em float32)
  meta.json         data dos dados e data de geração

Funções puras: window, rates_window, stock_tickers. I/O em main().
"""
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (CLEAN, FEATURES, MARKET, SECTOR_PROFILE, SECTORS, SNAPSHOT,  # noqa: E402
                             SNAPSHOT_FEATURES, SNAPSHOT_RATE_MARGIN_DAYS, SNAPSHOT_YEARS)
from src.models import cockpit as ck  # noqa: E402


def stock_tickers(profile: dict = SECTOR_PROFILE) -> list[str]:
    """Ações do microscópio, sem repetições, pela ordem dos setores."""
    return list(dict.fromkeys(a for p in profile.values() for a in p["acoes"]))


def window(prices: pd.DataFrame, years: int = SNAPSHOT_YEARS) -> pd.DataFrame:
    """Últimos `years` anos de calendário (mesma regra que o período '3y' do Yahoo)."""
    end = prices.index.max()
    return prices[prices.index > end - pd.DateOffset(years=years)]


def rates_window(rates: pd.Series, first_price: pd.Timestamp, margin_days: int = SNAPSHOT_RATE_MARGIN_DAYS) -> pd.Series:
    return rates.dropna()[rates.dropna().index >= first_price - pd.Timedelta(days=margin_days)]


def merge_stocks(new: pd.DataFrame, old: pd.DataFrame | None, stocks: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Ações novas onde vieram; as que falharam no download vêm do snapshot anterior. Devolve também a lista
    das que vieram do anterior."""
    falta = [t for t in stocks if t not in new]
    parts = [new[[t for t in stocks if t in new]]]
    if falta and old is not None:
        parts.append(old[[t for t in falta if t in old]])
    return pd.concat(parts, axis=1).reindex(columns=stocks), falta


def main(offline: bool = False) -> Path:
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    etfs = window(pd.read_parquet(CLEAN / "prices.parquet")[list(SECTORS) + [MARKET]])
    stocks = stock_tickers()
    old = SNAPSHOT / "prices.parquet"
    try:
        if offline:
            raise RuntimeError("modo offline")
        novas = ck.download_close(stocks, f"{SNAPSHOT_YEARS}y", allow_missing=True)
        acoes, falta = merge_stocks(novas, pd.read_parquet(old) if old.exists() else None, stocks)
        if falta:
            print(f"[snapshot] {len(falta)} ações sem dados novos, mantidas do snapshot anterior: {falta}")
    except Exception as e:  # noqa: BLE001  sem rede: mantém as ações do snapshot anterior
        if not old.exists():
            raise RuntimeError(f"Sem preços das ações e sem snapshot anterior ({e})") from e
        acoes = pd.read_parquet(old)[stocks]
        print(f"[snapshot] ações do snapshot anterior ({e})")
    prices = etfs.join(acoes, how="left")
    prices.astype("float32").to_parquet(SNAPSHOT / "prices.parquet", compression="zstd")

    rates = rates_window(pd.read_parquet(CLEAN / "factors.parquet")["juros"], prices.index.min())
    rates.rename("DGS10").to_frame().to_parquet(SNAPSHOT / "rates.parquet", compression="zstd")

    for f in SNAPSHOT_FEATURES:
        if f.endswith(".parquet"):
            d = pd.read_parquet(FEATURES / f)
            d.astype({c: "float32" for c in d.select_dtypes("float").columns}).to_parquet(
                SNAPSHOT / f, compression="zstd", index=False)
        else:
            shutil.copyfile(FEATURES / f, SNAPSHOT / f)
    meta = {"dados_ate": str(etfs.index.max().date()), "acoes_ate": str(acoes.index.max().date()),
            "juros_ate": str(rates.index.max().date()),
            "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (SNAPSHOT / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(p.stat().st_size for p in SNAPSHOT.iterdir()) / 1e6
    print(f"[snapshot] {SNAPSHOT} ({size:.1f} MB), dados até {meta['dados_ate']}")
    return SNAPSHOT


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
