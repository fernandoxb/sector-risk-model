"""Corre o pipeline de ponta a ponta: coleta → limpeza → modelação → backtesting do VaR → visão conjunta → scorecard → ligações fator → setor → dependência de cauda.

Uso:  python run_pipeline.py            (tudo)
      python run_pipeline.py --offline  (salta o download e usa data/raw existente: preços e fatores)
"""
import sys

from src.clean import validate
from src.collect import download_factors, download_prices, empresas
from src.models import backtest, backtest_empresas, cauda, decomposicao, factor_links, portfolio, risk_metrics, scorecard

if __name__ == "__main__":
    if "--offline" not in sys.argv:
        download_prices.main()
        download_factors.main()
        empresas.main()
        empresas.main_prices()
    validate.main()
    risk_metrics.main()
    backtest.main()
    backtest_empresas.main()
    portfolio.main()
    scorecard.main()
    factor_links.main()
    decomposicao.main()
    cauda.main()
