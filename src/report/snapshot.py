"""Snapshot do cockpit (app/cockpit/snapshot/): ficheiros pequenos e versionados para correr sem rede.

O cockpit tenta sempre os dados ao vivo (Yahoo e FRED). Se falharem, ou com COCKPIT_SNAPSHOT=1
(Streamlit Community Cloud), lê daqui. Conteúdo:
  prices.parquet    fechos ajustados dos 11 setores, SPY e ações do microscópio (últimos SNAPSHOT_YEARS anos)
  rates.parquet     Treasury 10Y (DGS10) desde SNAPSHOT_RATE_MARGIN_DAYS dias antes do 1.º preço
  <features>        cópias de data/features usadas pelo cockpit (séries do backtest em float32)
  companies.json    ficha de bolsa e últimas notícias de cada empresa (à data do run)
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
from config.settings import (CLEAN, FEATURES, MARKET, SECTORS, SNAPSHOT,  # noqa: E402
                             SNAPSHOT_FEATURES, SNAPSHOT_RATE_MARGIN_DAYS, SNAPSHOT_YEARS)
from src.collect.empresas import fetch_info, fetch_news, load_holdings  # noqa: E402
from src.models import cockpit as ck  # noqa: E402


def stock_tickers(profile: dict | None = None) -> list[str]:
    """Empresas do mapa de risco, sem repetições, pela ordem dos setores: as de holdings.csv ou, com `profile`
    (ou sem ficheiro), as de SECTOR_PROFILE."""
    if profile is not None:
        return list(dict.fromkeys(a for p in profile.values() for a in p["acoes"]))
    return list(dict.fromkeys(load_holdings()["ticker"]))


def companies_payload(tickers: list[str], antigo: dict, fetch_info, fetch_news) -> tuple[dict, list[str]]:
    """Ficha e notícias de cada empresa para o snapshot. Uma empresa que falhe mantém o que havia em `antigo`.
    Devolve também a lista das que falharam."""
    out, falhas = {}, []
    for t in tickers:
        try:
            out[t] = {"info": fetch_info(t), "noticias": fetch_news(t)}
        except Exception:  # noqa: BLE001
            falhas.append(t)
            if t in antigo:
                out[t] = antigo[t]
    return out, falhas


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

    emp_f = SNAPSHOT / "companies.json"
    antigo = json.loads(emp_f.read_text()) if emp_f.exists() else {}
    if offline:
        emp, falhas = {t: antigo[t] for t in stocks if t in antigo}, []
    else:
        emp, falhas = companies_payload(stocks, antigo, fetch_info, fetch_news)
        if falhas:
            print(f"[snapshot] {len(falhas)} empresas sem ficha nova: {falhas[:8]}{'…' if len(falhas) > 8 else ''}")
    emp_f.write_text(json.dumps(emp, ensure_ascii=False, separators=(",", ":"), default=str))

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
