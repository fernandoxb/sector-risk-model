"""Corre o pipeline de ponta a ponta: coleta → limpeza → modelação → backtesting do VaR → visão conjunta → scorecard → mapa.

Uso:  python run_pipeline.py            (tudo)
      python run_pipeline.py --offline  (salta o download e usa data/raw existente)
"""
import sys

from src.clean import validate
from src.collect import download_prices
from src.models import backtest, portfolio, risk_metrics, scorecard
from src.report import build_map

if __name__ == "__main__":
    if "--offline" not in sys.argv:
        download_prices.main()
    validate.main()
    risk_metrics.main()
    backtest.main()
    portfolio.main()
    scorecard.main()
    build_map.main()
