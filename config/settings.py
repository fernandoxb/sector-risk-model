"""Configuração central do modelo de risco setorial."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW, CLEAN, FEATURES = DATA / "raw", DATA / "clean", DATA / "features"

MARKET = "SPY"
SECTORS = {
    "XLK": "Tecnologia", "XLF": "Financeiro", "XLE": "Energia", "XLV": "Saúde",
    "XLI": "Industriais", "XLY": "Consumo discricionário", "XLP": "Consumo básico",
    "XLU": "Utilities", "XLB": "Materiais", "XLRE": "Imobiliário", "XLC": "Comunicações",
}
TICKERS = list(SECTORS) + [MARKET]
START = "2005-01-01"

TRADING_DAYS = 252
WIN_SHORT = 63          # 1 trimestre: regime de volatilidade atual
WIN_YEAR = 252          # 1 ano
WIN_VAR = 756           # 3 anos para VaR históricos e beta (em dias úteis)

# --- Risco de cauda: parâmetros únicos para o mapa, o cockpit e os relatórios -
# 97,5% é o nível do Expected Shortfall no FRTB (Basileia III); retornos simples, perda positiva.
CVAR_LEVEL = 0.975
CVAR_YEARS = 3          # janela histórica em anos de calendário
PANIC_Q = 0.20          # dias de pânico = 20% piores dias do SPY (downside beta)
MAX_ABS_RET = 0.25      # retornos diários acima disto vão para revisão

# --- Backtesting do VaR (src/models/backtest.py) -----------------------------
# VaR a 1 dia previsto só com dados até t-1, em janela móvel de CVAR_YEARS anos (em dias úteis).
BT_LEVELS = (0.99, 0.975)            # 99% = VaR regulamentar (Basileia); 97,5% = nível do ES no FRTB
BT_WINDOW = CVAR_YEARS * TRADING_DAYS
# histórico, GJR-GARCH(1,1)-t e Filtered Historical Simulation (mesmo GARCH, quantil empírico dos resíduos)
BT_METHODS = ("historico", "garch", "fhs")
BT_ES_GRID = 400                     # pontos da quadratura do ES da t-Student (por dia)
BT_ES_NSIM = 1000                    # simulações sob H0 para o p-valor do teste Z2 de Acerbi–Székely
BT_SEED = 42                         # semente das simulações (resultados reprodutíveis)
# FRTB, backtest por mesa: reprovada se, nos últimos 250 dias, houver mais exceções do que isto
BT_FRTB_MAX = {0.99: 12, 0.975: 30}
BT_REFIT_FREQ = "M"                  # re-estimar o GARCH no 1.º dia útil de cada mês (período pandas)
BT_SIGNIF = 0.05                     # nível de significância dos testes de Kupiec e Christoffersen
# Semáforo de Basileia: exceções nos últimos 250 dias. As zonas vêm da binomial acumulada
# P(X <= x | n, 1 - nível): verde se < 95%, vermelho se >= 99,99% (a 99%: verde 0-4, amarelo 5-9, vermelho 10+).
BT_TL_DAYS = 250
BT_TL_GREEN = 0.95
BT_TL_RED = 0.9999

# --- Cockpit interativo (app/cockpit) ---------------------------------------
# Classificação do setor para a cor do nó e as ações mais líquidas para o drill-down.
SECTOR_PROFILE = {
    "XLK":  {"tipo": "Sensível",  "acoes": ["AAPL", "MSFT", "NVDA", "AVGO"]},
    "XLF":  {"tipo": "Cíclico",   "acoes": ["JPM", "BAC", "MS", "GS"]},
    "XLV":  {"tipo": "Defensivo", "acoes": ["LLY", "UNH", "JNJ", "ABBV"]},
    "XLY":  {"tipo": "Cíclico",   "acoes": ["AMZN", "TSLA", "HD", "MCD"]},
    "XLP":  {"tipo": "Defensivo", "acoes": ["PG", "COST", "KO", "PEP"]},
    "XLE":  {"tipo": "Sensível",  "acoes": ["XOM", "CVX", "COP", "SLB"]},
    "XLI":  {"tipo": "Cíclico",   "acoes": ["GE", "CAT", "UNP", "HON"]},
    "XLU":  {"tipo": "Defensivo", "acoes": ["NEE", "SO", "DUK", "CEG"]},
    "XLB":  {"tipo": "Cíclico",   "acoes": ["LIN", "APD", "SHW", "FCX"]},
    "XLRE": {"tipo": "Cíclico",   "acoes": ["PLD", "AMT", "EQIX", "CCI"]},
    "XLC":  {"tipo": "Sensível",  "acoes": ["META", "GOOGL", "NFLX", "TMUS"]},
}
# Setores de crescimento (valuation de longa duração) penalizados quando os juros sobem.
GROWTH_SECTORS = ["XLK", "XLC", "XLY"]
RATE_SERIES = "DGS10"   # Treasury a 10 anos (FRED)
RATE_LOOKBACK = 63      # variação dos juros nos últimos ~3 meses define o regime
# Cofre Obsidian (Documents/Obsidian Vault). Pode ser alterado no painel ou via a variável OBSIDIAN_VAULT.
OBSIDIAN_VAULT = Path(os.environ.get("OBSIDIAN_VAULT", ROOT.parents[1] / "Obsidian Vault"))
OBSIDIAN_FOLDER = "Risco Setorial"
