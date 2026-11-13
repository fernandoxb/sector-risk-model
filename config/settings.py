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
BT_METHODS = ("historico", "garch", "fhs", "evt")
# EVT condicional (McNeil e Frey 2000; QRM 7.2.6): o mesmo GARCH, com uma distribuição de Pareto generalizada
# ajustada à cauda dos resíduos. O limiar é o resíduo que deixa EVT_TAIL_FRAC da amostra na cauda (no livro,
# 100 em 1000); com a janela de 756 dias são cerca de 75 excessos.
EVT_TAIL_FRAC = 0.10
EVT_XI_MAX = 0.9                     # ξ estimado é limitado (ξ >= 1 daria ES infinito)
BT_MF_NBOOT = 2000                   # reamostragens do teste de McNeil–Frey aos resíduos das exceções
BT_ES_GRID = 400                     # pontos da quadratura do ES da t-Student (por dia)
BT_ES_NSIM = 1000                    # simulações sob H0 para o p-valor do teste Z2 de Acerbi–Székely
BT_SEED = 42                         # semente das simulações (resultados reprodutíveis)
# FRTB, backtest por mesa: reprovada se, nos últimos 250 dias, houver mais exceções do que isto
BT_FRTB_MAX = {0.99: 12, 0.975: 30}

# Validação fora da amostra da escolha do método: escolhe-se com as previsões até BT_SPLIT (exclusive) e
# confirma-se com as seguintes, que não entraram na escolha. Um setor só conta num período com BT_SPLIT_MIN_DAYS.
BT_SPLIT = "2019-01-01"
BT_SPLIT_MIN_DAYS = 500

# --- Modelo oficial de VaR/ES (previsão para o próximo dia útil) -------------
# Escolhido pela regra backtest.choose_method com as previsões de 2008–2018 e confirmado em 2019–2026 e nas
# empresas (setores que passam Kupiec, Christoffersen e Z2 a 97,5%; dados até 2026-11-11):
#              escolha 2008–2018   confirmação 2019–2026   empresas (de 109)
#   evt             9 de 9              10 de 11                 75
#   fhs             6 de 9               9 de 11                 61
#   garch (t)       0 de 9               0 de 11                 22
#   historico       0 de 9               3 de 11                 21
# Até 2026-11-12 o oficial era "fhs" (ver docs/nota-metodologica.md, secção 15).
OFFICIAL_VAR_METHOD = "evt"
# --- Horizontes mais longos (src/models/horizonte.py, Fase E) -----------------
# VaR e ES a h dias úteis por simulação de trajetórias do GARCH (QRM 2.3.4 e McNeil–Frey 2000): 10 dias = horizonte
# regulamentar de Basileia; 21 dias = um mês.
HORIZONS = (10, 21)
HORIZON_NSIM = 20_000            # trajetórias da previsão de hoje
HORIZON_BT_NSIM = 2_000          # trajetórias por bloco no backtest (blocos de h dias sem sobreposição)

# Colunas de data/features/official_risk.csv: nome → (medida, nível)
OFFICIAL_MEASURES = {"var99": ("var", 0.99), "var975": ("var", 0.975), "es975": ("es", 0.975)}

# --- Scorecard de risco por setor, 0–100 (src/models/scorecard.py) ----------
# Cada componente vai para [0, 1] pelo percentil entre os 11 setores (regime já está em [0, 1]).
# A cauda prevista pesa mais por ser a única medida forward-looking validada pelo backtest.
SCORE_WEIGHTS = {"cauda": 0.30, "sistemico": 0.20, "stress": 0.20, "regime": 0.15, "momento": 0.15}
SCORE_IMPACT = ("cauda", "sistemico", "stress")      # → impacto 1–5 no mapa (P × I)
SCORE_PROBABILITY = ("regime", "momento")            # → probabilidade 1–5 no mapa
STRESS_ONLY_LOSSES = True      # um ganho num cenário (XLE em 2022: +44,5%) conta como perda zero, não compensa as outras
STRESS_PROXY_WEIGHT = 0.5      # cenários com proxy β × SPY pesam metade na perda média de stress
MOMENTUM_DAYS = 60             # janela de exceções ao VaR 97,5% oficial
# Risco de modelo: semáforo de Basileia (definido a 99%) do método oficial → multiplicador do score.
MODEL_RISK_LEVEL = 0.99
MODEL_RISK_MULT = {"verde": 1.00, "amarelo": 1.10, "vermelho": 1.25}
# Cauda: metade pela posição entre os 11 setores, metade pelo percentil do ES de hoje na história do próprio setor
# (as previsões do backtest), para um score baixo não querer dizer só "o menos mau dos 11".
SCORE_HIST_WEIGHT = 0.5
# Incerteza do ranking: pesos sorteados à volta dos oficiais, Dirichlet(SCORE_RANK_CONC · peso); com 20, o peso de
# 0,30 varia com desvio-padrão de cerca de 0,10.
SCORE_RANK_NSIM = 5000
SCORE_RANK_CONC = 20.0
SCORE_SENSITIVITY = 0.5        # análise de sensibilidade: cada peso varia ±50% (e re-normaliza)

# --- Ligações medidas fator macro → setor (src/models/factor_links.py) ------
# id do nó no mapa → série, fonte, transformação semanal e direção adversa (+1 = subir é adverso).
#   transf "diff": variação em p.p. (séries do FRED em %); "logdiff": variação log do nível.
MACRO_FACTORS = {
    "juros":    {"serie": "DGS10",        "fonte": "fred",  "transf": "diff",    "adverso": +1,
                 "nome": "Treasury 10Y", "curto": "Juros 10Y"},
    "inflacao": {"serie": "T10YIE",       "fonte": "fred",  "transf": "diff",    "adverso": +1,
                 "nome": "Breakeven de inflação 10Y", "curto": "Inflação implícita"},
    # O FRED só dá o spread high yield da ICE para os últimos 3 anos (licença): chega para as ligações (CVAR_YEARS).
    # Onde é preciso mais história (simulador de choques, 10 anos, e o teste nas crises desde 2008) usa-se a série
    # longa: o spread Baa (Moody's) − Treasury 10Y, desde 1986, na coluna "spreads_longo".
    "spreads":  {"serie": "BAMLH0A0HYM2", "fonte": "fred",  "transf": "diff",    "adverso": +1,
                 "nome": "Spread high yield (ICE BofA)", "curto": "Spread high yield",
                 "serie_longa": "BAA10Y", "curto_longo": "Spread de crédito Baa"},
    "vix":      {"serie": "^VIX",         "fonte": "yahoo", "transf": "logdiff", "adverso": +1,
                 "nome": "VIX", "curto": "VIX"},
    "petroleo": {"serie": "CL=F",         "fonte": "yahoo", "transf": "logdiff", "adverso": +1,
                 "nome": "Petróleo WTI (futuro)", "curto": "Petróleo"},
    "dolar":    {"serie": "DX-Y.NYB",     "fonte": "yahoo", "transf": "logdiff", "adverso": +1,
                 "nome": "Índice do dólar (DXY)", "curto": "Dólar"},
}
FACTOR_START = START          # histórico descarregado dos fatores (as ligações usam CVAR_YEARS; o resto, as janelas longas)
LINK_FREQ = "W-FRI"           # retornos semanais, sexta a sexta
LINK_T_MIN = 2.0              # |t| HAC mínimo para aceitar uma ligação
LINK_HAC_LAGS = None          # None = regra de Newey–West: floor(4 · (T/100)^(2/9))
# Regressão conjunta (os 6 fatores ao mesmo tempo, variáveis padronizadas): separa fatores correlacionados
# (VIX e spreads). Ridge com penalização FACTOR_RIDGE_ALPHA; uma ligação resiste se |β padronizado| >=
# FACTOR_BETA_MIN e |t| HAC da regressão conjunta >= LINK_T_MIN.
FACTOR_RIDGE_ALPHA = 1.0
FACTOR_BETA_MIN = 0.05

# --- Decomposição do risco de cada empresa (src/models/decomposicao.py) ------
DECOMP_MIN_DAYS = TRADING_DAYS       # dias comuns mínimos (empresa, setor e mercado) na janela CVAR_YEARS

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
# (A antiga lista GROWTH_SECTORS saiu: o efeito dos juros usa o β medido em factor_links.csv, ver cockpit.rate_stress.)
RATE_SERIES = "DGS10"   # Treasury a 10 anos (FRED)
RATE_LOOKBACK = 63      # variação dos juros nos últimos ~3 meses define o regime
RATE_THRESHOLD_PP = 0.25  # subida/descida se a variação em RATE_LOOKBACK dias for >= ±0,25 p.p.; senão "estável"
# Cofre Obsidian (Documents/Obsidian Vault). Pode ser alterado no painel ou via a variável OBSIDIAN_VAULT.
OBSIDIAN_VAULT = Path(os.environ.get("OBSIDIAN_VAULT", ROOT.parents[1] / "Obsidian Vault"))
OBSIDIAN_FOLDER = "Risco Setorial"

# --- Bots e alertas (flows/pipeline.py, src/report/alerts.py, src/report/weekly.py) ------
STATE = DATA / "state"               # um estado por data de dados (estado_AAAA-MM-DD.json) para comparar com o anterior
LOGS = ROOT / "logs"                 # saída do launchd e do flow
COLLECT_RETRIES = 2                  # novas tentativas da coleta (rede) antes de o flow falhar
COLLECT_RETRY_DELAY = 300            # segundos entre tentativas
# Dados atrasados (o download correu mas os preços não chegam à última sessão fechada): o Coletor repete o
# download até COLLECT_STALE_TRIES vezes, com COLLECT_STALE_WAIT segundos entre elas; se continuar atrasado,
# o flow segue e o Sentinela dá o alerta "dados atrasados".
COLLECT_STALE_TRIES = 3
COLLECT_STALE_WAIT = 600
# Níveis do score de risco P × I (1–25): limite superior de cada nível, do mais baixo para o mais alto.
PI_LEVELS = ((4, "baixo"), (9, "moderado"), (16, "elevado"), (25, "crítico"))
SEMAFORO_ORDEM = ("verde", "amarelo", "vermelho")
ALERT_SEVERITIES = ("alta", "média", "baixa")
ALERT_NOTIFY = ("alta",)             # só estas severidades geram notificação macOS
ALERT_KEEP_DAYS = 60                 # secções de dias mantidas em Alertas.md (as mais antigas saem)
WEEKLY_TOP = 3                       # setores no top do relatório semanal

# --- Montra: modo snapshot do cockpit (app/cockpit/snapshot, versionado) -------
# Usado quando não há rede ou o Yahoo limita (ex.: Streamlit Community Cloud). Atualizado pelo flow.
SNAPSHOT = ROOT / "app" / "cockpit" / "snapshot"
SNAPSHOT_YEARS = 10             # anos de preços guardados (a janela mais longa do cockpit)
SNAPSHOT_RATE_MARGIN_DAYS = 200 # dias de juros antes do 1.º preço (o regime usa RATE_LOOKBACK dias úteis)
# Ficheiros de data/features copiados tal como estão (o cockpit lê-os do snapshot se data/ não existir).
SNAPSHOT_FEATURES = ("scorecard.csv", "official_risk.csv", "var_backtest.csv", "var_backtest_series.parquet",
                     "risk_contribution.csv", "stress_tests.csv", "covariance_shrinkage.csv", "factor_links.csv",
                     "alerts.json", "sector_metrics.csv", "score_history.csv", "holdings.csv",
                     "company_backtest.csv", "company_risk.csv", "var_backtest_split.csv",
                     "var_backtest_multiple.csv", "tail_dependence.csv", "tail_meta.json", "tail_dependence_janelas.csv",
                     "company_factors.csv", "tail_scenarios.csv", "cenario_semanal.parquet", "cenario_validacao.csv",
                     "horizon_risk.csv", "horizon_backtest.csv")

# --- Cenário de stress próprio (cockpit, separador Carteira e stress) --------
# Choque instantâneo propagado pelos betas: mercado pelo beta diário de 3 anos (sector_metrics.csv) e cada
# fator macro pelo β semanal da regressão HAC (factor_links.csv, só ligações aceites). Um fator de cada vez.
# Limites dos sliders: (mínimo, máximo, padrão, passo), em % para o SPY e fatores "logdiff", em p.p. para "diff".
SHOCK_MARKET = (-40.0, 20.0, -10.0, 1.0)

# --- Motor de cenários (src/models/cenarios.py, Fase 3) ---------------------
# Choque nos fatores → outros fatores (normal condicional) → setores (regressão nos seis fatores + resíduos
# reamostrados) → empresas (beta ao seu ETF + resíduos), por Monte Carlo, em semanas.
CENARIO_ANOS = 10                       # janela de estimação (inclui a COVID-19 e 2022)
CENARIO_NSIM = 5000                     # cenários por simulação
CENARIO_HORIZONTES = {"1 semana": 1, "1 mês": 4, "3 meses": 13}
# Validação fora da amostra: o modelo estimado só com as semanas anteriores recebe as variações reais dos fatores
# (sextas-feiras; o episódio são as semanas em (início, fim]).
CENARIO_EPISODIOS = {"2008 · Lehman": ("2008-09-12", "2008-11-21"),
                     "2020 · COVID-19": ("2020-02-21", "2020-03-20"),
                     "2022 · Subida de juros": ("2021-12-31", "2022-06-17")}
SHOCK_DIFF = (-2.0, 3.0, 1.0, 0.25)
SHOCK_LOGDIFF = (-50.0, 150.0, 50.0, 5.0)


# --- Calendário do mercado (cache do Coletor e alerta de dados atrasados) -----
# Uma sessão conta como fechada 1 h depois do fecho de Nova Iorque (16:00), para os preços do Yahoo assentarem
# (17:00 em NY = 22:00 em Lisboa, antes do run agendado das 22:30).
# Feriados da NYSE: calculados pelas regras da bolsa em src/collect/sessao.py (sem dependências), mais os
# fechos excecionais abaixo, que nenhuma regra prevê.
MARKET_TZ = "America/New_York"
MARKET_CLOSE = "17:00"
# Fechos antecipados (13:00 em NY: 3 de julho, dia a seguir ao Thanksgiving, véspera de Natal): a sessão conta
# como fechada às 14:00, com a mesma margem de 1 h.
MARKET_EARLY_CLOSE = "14:00"
CALENDAR_PREVIEW = 3                 # próximos feriados mostrados no cockpit
NYSE_JUNETEENTH_FROM = 2022          # o Juneteenth é feriado da NYSE desde 2022
NYSE_SPECIAL_CLOSURES = {
    "2007-01-02": "luto nacional (Gerald Ford)",
    "2012-10-29": "furacão Sandy",
    "2012-10-30": "furacão Sandy",
    "2018-12-05": "luto nacional (George H. W. Bush)",
    "2025-01-09": "luto nacional (Jimmy Carter)",
}

# --- Publicação semanal do snapshot (bot Publicador, src/report/publicar.py) --
# O pipeline grava o snapshot em SNAPSHOT_LIVE (fora do Git). Na última sessão de cada semana, o Publicador envia-o
# para o ramo SNAPSHOT_PUBLISH_BRANCH como um único commit sem pai, que substitui o anterior: o `main` só tem código
# e o repositório não cresce todas as semanas. O cockpit no Streamlit Cloud lê esse ramo (SNAPSHOT_REMOTE_URL); se
# não conseguir, usa a cópia embutida em SNAPSHOT, que também serve os testes e só muda quando se corre
# `python -m src.report.snapshot --embutido`.
SNAPSHOT_LIVE = DATA / "snapshot"
SNAPSHOT_PUBLISH = True
SNAPSHOT_PUBLISH_BRANCH = "snapshot"
SNAPSHOT_PUBLISH_REMOTE = "origin"
SNAPSHOT_REMOTE_URL = "https://raw.githubusercontent.com/fernandoxb/sector-risk-model/snapshot"
SNAPSHOT_REMOTE_TTL = 6 * 3600       # segundos entre leituras do ramo pelo cockpit
SNAPSHOT_REMOTE_TIMEOUT = 10         # segundos por ficheiro
SNAPSHOT_FILES = ("meta.json", "prices.parquet", "rates.parquet", "companies.json") + SNAPSHOT_FEATURES


# --- Empresas de cada setor (mapa de risco: setor → empresas → ficha) ---------
# As HOLDINGS_TOP maiores posições de cada ETF, com o peso no ETF (Yahoo, fundos). Se o Yahoo falhar, usa-se o
# ficheiro anterior e, na falta dele, as ações de SECTOR_PROFILE com pesos iguais.
HOLDINGS_TOP = 10
COMPANY_NEWS = 6                # notícias mostradas na ficha da empresa
COMPANY_TTL = 900               # segundos de cache da ficha e das notícias ao vivo no cockpit

# --- Carteira do utilizador (cockpit, separador Minha carteira) ----------------
# Ativos permitidos: as empresas de holdings.csv, os 11 ETFs setoriais e o SPY. A carteira guarda-se no link
# (?carteira=AAPL:30,MSFT:20,...), sem contas de utilizador.
USER_PORTFOLIO_MAX = 25         # máximo de posições
USER_PORTFOLIO_DEFAULT = 6      # carteira de exemplo: a maior posição dos primeiros N setores, pesos iguais
USER_MARKET_LABEL = "Mercado"   # "setor" do SPY na exposição setorial
# Ações fora das 110 do mapa: o cockpit valida qualquer ticker no Yahoo e põe-no no ETF do seu setor (classificação
# do Yahoo). Sem setor conhecido (ex.: um ETF de ouro), liga-se ao mercado. As de EXTRA_TICKERS entram também no
# pipeline diário (preços, modelo e backtest, como as do mapa), com peso 0 no ETF.
YAHOO_SETOR_ETF = {"Technology": "XLK", "Financial Services": "XLF", "Energy": "XLE", "Healthcare": "XLV",
                   "Industrials": "XLI", "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Utilities": "XLU",
                   "Basic Materials": "XLB", "Real Estate": "XLRE", "Communication Services": "XLC"}
EXTRA_TICKERS: tuple[str, ...] = ()
# Carteira alternativa por Hierarchical Risk Parity (López de Prado, 2016), com a distância das quedas conjuntas
HRP_RISK_MEASURE = "cvar"       # risco de cada ramo: "cvar" (CVAR_LEVEL, histórico) ou "variance" (Ledoit-Wolf)
HRP_MIN_DAYS = 252              # dias comuns mínimos para calcular
HRP_LINKAGE = "single"          # método de agrupamento (scipy.cluster.hierarchy.linkage)
# Teste fora da amostra: pesos HRP recalculados a cada HRP_OOS_PASSO dias só com os HRP_OOS_JANELA dias anteriores
HRP_OOS_JANELA = 2 * TRADING_DAYS
HRP_OOS_PASSO = 63              # cerca de 3 meses

# --- Modelo de risco por empresa (src/models/backtest_empresas.py) -----------
# O mesmo backtest dos setores (histórico, GARCH-t, FHS) aplicado às empresas de holdings.csv, com preços
# desde COMPANY_BT_START (2005, como os setores; até 2026-11-13, 2015). Com a janela de BT_WINDOW dias, as previsões
# começam cerca de 3 anos depois, em 2008. O backtest completo demora cerca de 8 minutos (só na última sessão da semana).
COMPANY_START = START
COMPANY_BT_START = START
COMPANY_MIN_DAYS = BT_WINDOW + 60    # histórico mínimo para ter modelo (senão a empresa fica só com risco descritivo)
# Previsão FHS da carteira do utilizador no cockpit: GARCH sobre os retornos da carteira com os pesos atuais.
USER_FORECAST_YEARS = 5              # anos de preços pedidos ao Yahoo (precisa de BT_WINDOW dias + 1 mês)
# Correção do ES subestimado: quando o Z2 rejeita com Z2 < 0, o ES previsto é multiplicado por k = 1 − Z2, o
# fator que teria posto o Z2 a zero no backtest (Z2 = 1 − Σ L·I / (T·p·ES) é linear em 1/ES). Limitado a ES_ADJ_MAX.
ES_ADJ_MAX = 1.5
# Run diário: o backtest completo das empresas só corre na última sessão de cada semana (ou se faltar o ficheiro,
# ou se as empresas mudarem); nos outros dias calcula-se só a previsão do dia seguinte (uma estimação por empresa).
COMPANY_FULL_WEEKLY = True


# --- Dependência de cauda (src/models/cauda.py; McNeil, Frey e Embrechts 2005, cap. 5) ----------
# "Quando um cai muito, o outro cai também?" Duas medidas, na janela de CVAR_YEARS anos:
#   p_conj  frequência observada: P(B nos CAUDA_Q piores dias | A nos CAUDA_Q piores dias), em retornos simples
#   lam     coeficiente de dependência de cauda λ da cópula t ajustada aos resíduos do GARCH (ρ pelo tau de
#           Kendall, ν por máxima verosimilhança)
CAUDA_Q = 0.05                  # cauda = 5% piores dias (cerca de 38 dias em 3 anos)
CAUDA_NU_BOUNDS = (2.1, 60.0)   # limites dos graus de liberdade ν da cópula t (ν alto ≈ cópula de Gauss, λ ≈ 0)
# Janelas do mapa de risco (anos → cauda q). Com 1 ano, os 5% piores dias seriam só cerca de 12: usa-se 10%.
# Uma série entra numa janela se tiver dados em pelo menos CAUDA_MIN_COBERTURA dos dias (o XLC só existe desde 2018).
CAUDA_JANELAS = {1: 0.10, 3: CAUDA_Q, 5: CAUDA_Q, 10: CAUDA_Q}
CAUDA_MIN_COBERTURA = 0.8
# Cenários conjuntos: cópula t assimétrica (skewed t, QRM 5.1.? e McNeil, Frey e Embrechts 2005, secção 3.2.3),
# X = γW + √W·Z, com W ~ IG(ν/2, ν/2). γ < 0 dá mais quedas conjuntas do que subidas conjuntas. γ é escolhido nesta
# grelha para a simulação acertar ao mesmo tempo as médias observadas de quedas e de subidas conjuntas.
CAUDA_GAMA_GRELHA = tuple(round(-0.025 * k, 3) for k in range(25))       # 0 a −0,6
CAUDA_GAMA_NSIM = 50_000                                                  # dias simulados por valor da grelha
CAUDA_NSIM = 200_000            # dias simulados da cópula t para os cenários conjuntos (cerca de 10 000 por cenário)
