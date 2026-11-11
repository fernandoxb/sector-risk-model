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
- Modelo oficial de VaR/ES (forward-looking): `OFFICIAL_VAR_METHOD = "fhs"`, escolhido pelo backtest;
  previsão de amanhã em `data/features/official_risk.csv`. O CVaR histórico continua a ser o do mapa e do cockpit.
- Rótulos na UI dizem sempre nível e janela (ex.: "CVaR 97,5% · 1 dia · 3 anos").

## Pipeline e estrutura
- `python run_pipeline.py` (ou `--offline` para não descarregar): coleta → validação → métricas → mapa.
  - `src/collect/` Coletor (yfinance, um pedido, `threads=False`; fatores macro do FRED + Yahoo) → `data/raw/`
  - `src/clean/` Inspetor (pandera) → `data/clean/` + `quality_report.md`
  - `src/models/` Modelador (`risk_metrics.py`, `cockpit.py`, `backtest.py`, `portfolio.py`, `scorecard.py`,
    `factor_links.py`) → `data/features/`
  - `src/report/` Repórter → embebe `data/features/risk_map_data.json` no mapa (`/*DATA*/…/*END*/`);
    Sentinela (`alerts.py`, estados em `data/state/`) e relatório semanal (`weekly.py`) → cofre Obsidian
- `flows/pipeline.py` — flow Prefect (uma task por bot); agendado com launchd (`flows/install_launchd.sh`).
- `app/risk-map/index.html` — mapa interativo (HTML autónomo, sem dependências externas de dados).
- `app/cockpit/app.py` — Streamlit: `.venv/bin/streamlit run app/cockpit/app.py`. Sem rede (ou com
  `COCKPIT_SNAPSHOT=1`, no Streamlit Cloud) lê `app/cockpit/snapshot/`, gerado por `src/report/snapshot.py`;
  dependências do Cloud em `app/cockpit/requirements.txt`.
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
