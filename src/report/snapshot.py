"""Snapshot do cockpit: os ficheiros de que precisa para correr sem o pipeline (Streamlit Cloud) ou sem rede.

Por omissão grava em data/snapshot/ (fora do Git), que o Publicador envia para o ramo `snapshot`. Com `--embutido`
atualiza app/cockpit/snapshot/, a cópia versionada que serve de recurso ao cockpit e de dados aos testes. Conteúdo:
  indices.parquet   índices base 100 dos 11 setores, SPY e empresas do mapa (últimos SNAPSHOT_YEARS anos): o cockpit
                    só usa retornos, e o repositório público não redistribui os preços do Yahoo
  rates.parquet     Treasury 10Y (DGS10) desde SNAPSHOT_RATE_MARGIN_DAYS dias antes do 1.º preço
  <features>        cópias de data/features usadas pelo cockpit (séries do backtest em float32)
  meta.json         data dos dados e data de geração

Funções puras: window, rates_window, stock_tickers, indices_base_100. I/O em main().
As fichas e notícias das empresas não vão no snapshot (texto do Yahoo): o cockpit lê-as ao vivo.
"""
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (CLEAN, FEATURES, MARKET, SECTORS, SNAPSHOT, SNAPSHOT_LIVE,  # noqa: E402
                             SNAPSHOT_FEATURES, SNAPSHOT_RATE_MARGIN_DAYS, SNAPSHOT_YEARS)
from src.collect.empresas import load_holdings  # noqa: E402


def stock_tickers(profile: dict | None = None) -> list[str]:
    """Empresas do mapa de risco, sem repetições, pela ordem dos setores: as de holdings.csv ou, com `profile`
    (ou sem ficheiro), as de SECTOR_PROFILE."""
    if profile is not None:
        return list(dict.fromkeys(a for p in profile.values() for a in p["acoes"]))
    return list(dict.fromkeys(load_holdings()["ticker"]))


def window(prices: pd.DataFrame, years: int = SNAPSHOT_YEARS) -> pd.DataFrame:
    """Últimos `years` anos de calendário (mesma regra que o período '3y' do Yahoo)."""
    end = prices.index.max()
    return prices[prices.index > end - pd.DateOffset(years=years)]


def rates_window(rates: pd.Series, first_price: pd.Timestamp, margin_days: int = SNAPSHOT_RATE_MARGIN_DAYS) -> pd.Series:
    return rates.dropna()[rates.dropna().index >= first_price - pd.Timedelta(days=margin_days)]


def indices_base_100(precos: pd.DataFrame) -> pd.DataFrame:
    """Cada série dividida pelo seu 1.º valor da janela × 100: os mesmos retornos, sem os níveis de preço."""
    return precos / precos.bfill().iloc[0] * 100


def main(dest: Path = SNAPSHOT_LIVE) -> Path:
    """Grava o snapshot em `dest`: por omissão SNAPSHOT_LIVE (fora do Git, o que o Publicador envia); com
    dest=SNAPSHOT atualiza a cópia embutida no repositório (recurso do cockpit e dados dos testes)."""
    dest.mkdir(parents=True, exist_ok=True)
    for velho in ("prices.parquet", "companies.json"):         # formatos antigos, com dados brutos do Yahoo
        (dest / velho).unlink(missing_ok=True)
    etfs = window(pd.read_parquet(CLEAN / "prices.parquet")[list(SECTORS) + [MARKET]])
    emp = pd.read_parquet(CLEAN / "prices_empresas.parquet")
    acoes = emp[[t for t in stock_tickers() if t in emp]]
    prices = etfs.join(acoes, how="left")
    indices_base_100(prices).astype("float32").to_parquet(dest / "indices.parquet", compression="zstd")

    rates = rates_window(pd.read_parquet(CLEAN / "factors.parquet")["juros"], prices.index.min())
    rates.rename("DGS10").to_frame().to_parquet(dest / "rates.parquet", compression="zstd")

    for f in SNAPSHOT_FEATURES:
        if f.endswith(".parquet"):
            d = pd.read_parquet(FEATURES / f)
            d.astype({c: "float32" for c in d.select_dtypes("float").columns}).to_parquet(
                dest / f, compression="zstd", index=False)
        else:
            shutil.copyfile(FEATURES / f, dest / f)
    meta = {"dados_ate": str(etfs.index.max().date()), "acoes_ate": str(acoes.dropna(how="all").index.max().date()),
            "juros_ate": str(rates.index.max().date()),
            "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (dest / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(p.stat().st_size for p in dest.iterdir()) / 1e6
    print(f"[snapshot] {dest} ({size:.1f} MB), dados até {meta['dados_ate']}")
    return dest


if __name__ == "__main__":
    main(dest=SNAPSHOT if "--embutido" in sys.argv else SNAPSHOT_LIVE)
