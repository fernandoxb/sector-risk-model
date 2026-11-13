# CLAUDE.md — Modelo de Risco Setorial

Projeto de portfólio (CV, mestrado em Risk Management): mede o risco dos 11 ETFs setoriais SPDR
(XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, XLC) face ao SPY, individualmente e em conjunto.
Idioma do código, comentários, UI e documentação: **português europeu**.

## Fonte única de verdade
- **Todos os parâmetros vivem em `config/settings.py`.** Nunca escrever números mágicos (níveis de confiança,
  janelas, limiares, tickers, caminhos) noutros ficheiros: importar de lá.
- **Uma função por métrica.** O CVaR é `src/models/cockpit.py::cvar`. O mapa, o cockpit e os relatórios usam
  todos essa função. Antes de criar uma métrica nova, procurar se já existe.

## Convenções de risco
- Risco de cauda (VaR, CVaR, beta, downside beta): **retornos simples**, **perdas em número positivo**.
- Volatilidade e GARCH: retornos logarítmicos.
- Carteira / visão conjunta: retornos simples (agregação linear nos pesos), também para a volatilidade.
- CVaR padrão: `CVAR_LEVEL = 0.975` (Expected Shortfall do FRTB, Basileia III), janela `CVAR_YEARS = 3`.
- Pânico (downside beta): `PANIC_Q = 0.20` (20% piores dias do SPY).
- Modelo oficial de VaR/ES (forward-looking): `OFFICIAL_VAR_METHOD = "evt"` (GARCH + cauda de Pareto), escolhido pelo backtest;
  previsão de amanhã em `data/features/official_risk.csv`. O CVaR histórico continua a ser o do mapa e do cockpit.
- Rótulos na UI dizem sempre nível e janela (ex.: "CVaR 97,5% · 1 dia · 3 anos").
- Números na UI em português, sempre via `src/report/formato.py` (vírgula decimal, "−" tipográfico); perdas de
  stress mostradas como retorno com sinal. Níveis P × I e "crítico" vêm de `PI_LEVELS` (`alerts.pi_level`/`pi_min`).

## Pipeline e estrutura
- `python run_pipeline.py` (ou `--offline` para não descarregar): coleta → validação → métricas.
  - `src/collect/` Coletor (yfinance, um pedido, `threads=False`; fatores macro do FRED + Yahoo) → `data/raw/`
  - `src/clean/` Inspetor (pandera) → `data/clean/` + `quality_report.md`
  - `src/models/` Modelador (`risk_metrics.py`, `cockpit.py`, `backtest.py`, `portfolio.py`, `scorecard.py`,
    `factor_links.py`) → `data/features/`
  - `src/report/` Sentinela (`alerts.py`, estados em `data/state/`) e relatório semanal (`weekly.py`) → cofre Obsidian
- `flows/pipeline.py` — flow Prefect (uma task por bot); agendado com launchd (`flows/install_launchd.sh`).
- `app/cockpit/app.py` — Streamlit: `.venv/bin/streamlit run app/cockpit/app.py`. O `app.py` carrega os dados e
  desenha a barra lateral; cada separador é `app/cockpit/paineis/<nome>.py::render(g)`, em que `g` são as variáveis
  do `app.py` nesse run; estilo e helpers de apresentação em `paineis/comum.py`. Sem rede (ou com
  `COCKPIT_SNAPSHOT=1`, no Streamlit Cloud) lê `app/cockpit/snapshot/`, gerado por `src/report/snapshot.py`;
  dependências do Cloud em `app/cockpit/requirements.txt`.
- Mapa de risco = separador do cockpit (setor → 10 maiores empresas → ficha com bolsa e notícias); empresas em
  `data/features/holdings.csv` (`src/collect/empresas.py`). O mapa HTML antigo foi apagado.
- Empresas: `backtest_empresas.py` reutiliza `backtest.run` (sem lógica nova) → `company_backtest.csv`, `company_risk.csv`
  (`validado`, `es_subestimado`). Carteira do utilizador: `carteira.py`; previsão = FHS sobre os retornos da carteira.
- Dependência de cauda (`src/models/cauda.py`, QRM cap. 5): `tail_dependence.csv` liga o mapa de risco por quedas
  conjuntas (padrão) ou correlação.
- Modelos de fatores (Fase C, QRM 3.4): `factor_links.fit_multivariate_ridge` junta colunas `*_multi` ao `factor_links.csv`
  (as antigas não mudam; Sentinela e choques usam as antigas); `decomposicao.py` → `company_factors.csv`.
- `scripts/` — capturas de ecrã (`capturas.py`) e PDF da nota (`nota_pdf.py`), via Chrome headless.
- Funções de cálculo são **puras** (sem Streamlit, sem I/O) para poderem ser testadas.
- `data/`, `reference/`, `.venv/` não vão para o Git.

## Obsidian
- Cofre: `OBSIDIAN_VAULT` (padrão `~/Documents/Obsidian Vault`), pasta `Risco Setorial/`.
- Notas com frontmatter YAML para o Dataview; tickers como links `[[XLK]]`.
- `00 Painel.md` agrega logs diários, `Semana AAAA-Www.md` e `Alertas.md` — manter os campos do frontmatter estáveis.

## Qualidade
- Validar cálculos novos contra uma implementação independente (ex.: numpy à mão vs biblioteca) antes de os usar.
- Testes em `tests/` com pytest; correr antes de cada commit.
- Commits pequenos e descritivos em português; não commitar dados nem segredos (`.env`, `FRED_API_KEY`).
- Atualizar `docs/nota-metodologica.md` quando mudar uma metodologia.

## Roteiro
1. ✅ Sprint 1 — pipeline, mapa de riscos com dados reais, cockpit.
2. ✅ Sprint 2 — backtesting do VaR (Kupiec, Christoffersen) e do ES (Z2), histórico vs GARCH-t vs FHS.
3. ✅ Sprint 3 — visão conjunta: Ledoit-Wolf, contribuição para o risco, stress tests (2008, 2020, 2022).
4. ✅ Sprint 4 — bots com Prefect (agendamento launchd), alertas do Sentinela e relatório semanal no Obsidian.
5. ✅ Sprint 5 — montra: README com resultados e capturas, nota metodológica em PDF, cockpit no Streamlit Cloud (modo snapshot).

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.
