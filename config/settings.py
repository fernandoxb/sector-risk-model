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

# --- Modelo oficial de VaR/ES (previsão para o próximo dia útil) -------------
# Escolhido pelo backtest de 2008–2026 (data/features/var_backtest.csv, p >= 5%, setores que passam em 11):
#              97,5%: Kupiec  Christoffersen  condicional  Z2  |  99%: Kupiec  Christoffersen  condicional  Z2
#   fhs                 11         10             10       10  |         10         8              8       10
#   garch (t)            1         10              2        0  |          1        11              1        0
#   historico            7          1              1        3  |          5         1              1        2
# O FHS é o único que passa a cobertura, a independência e o teste do ES ao mesmo tempo.
OFFICIAL_VAR_METHOD = "fhs"
# Colunas de data/features/official_risk.csv: nome → (medida, nível)
OFFICIAL_MEASURES = {"var99": ("var", 0.99), "var975": ("var", 0.975), "es975": ("es", 0.975)}

# --- Scorecard de risco por setor, 0–100 (src/models/scorecard.py) ----------
# Cada componente vai para [0, 1] pelo percentil entre os 11 setores (regime já está em [0, 1]).
# A cauda prevista pesa mais por ser a única medida forward-looking validada pelo backtest.
SCORE_WEIGHTS = {"cauda": 0.30, "sistemico": 0.20, "stress": 0.20, "regime": 0.15, "momento": 0.15}
SCORE_IMPACT = ("cauda", "sistemico", "stress")      # → impacto 1–5 no mapa (P × I)
SCORE_PROBABILITY = ("regime", "momento")            # → probabilidade 1–5 no mapa
STRESS_PROXY_WEIGHT = 0.5      # cenários com proxy β × SPY pesam metade na perda média de stress
MOMENTUM_DAYS = 60             # janela de exceções ao VaR 97,5% oficial
# Risco de modelo: semáforo de Basileia (definido a 99%) do método oficial → multiplicador do score.
MODEL_RISK_LEVEL = 0.99
MODEL_RISK_MULT = {"verde": 1.00, "amarelo": 1.10, "vermelho": 1.25}
SCORE_SENSITIVITY = 0.5        # análise de sensibilidade: cada peso varia ±50% (e re-normaliza)

# --- Visão conjunta (src/models/portfolio.py) --------------------------------
# Carteira de referência: None = pesos iguais nos 11 setores; ou um dict {ticker: peso} que soma 1.
PORTFOLIO_WEIGHTS: dict[str, float] | None = None
# Ledoit-Wolf vs covariância amostral: janelas de estimação (dias úteis) e re-balanceamento
# da carteira de variância mínima no teste fora da amostra.
LW_WINDOWS = (WIN_SHORT, WIN_VAR)
LW_REBAL_FREQ = "M"
# Stress tests históricos: pico → vale do S&P 500 (SPY ajustado), datas inclusivas.
STRESS_SCENARIOS = {
    "2008 · Crise financeira": ("2007-10-09", "2009-03-09"),
    "2020 · COVID-19": ("2020-02-19", "2020-03-23"),
    "2022 · Subida de juros": ("2022-01-03", "2022-10-12"),
}
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
