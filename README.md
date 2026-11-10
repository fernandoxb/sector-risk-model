# Modelo de Risco Setorial

Pipeline em Python que mede o risco dos 11 setores do S&P 500, individualmente e em conjunto.
Fases: coleta → limpeza → transformação → modelação → visualização.

## Estrutura
- `app/risk-map/` — mapa de riscos interativo (abre `index.html` no browser)
- `app/cockpit/` — cockpit Streamlit: CVaR, Downside Beta, rede MST, drill-down por ações, regime de juros e log para o Obsidian (`docs/nota-metodologica.md`)
- `src/` — código por fase (collect, clean, transform, models, report)
- `data/` — raw / clean / features (não vai para o Git)
- `reference/` — repositórios clonados para consulta (exemplos e notebooks; não vai para o Git)
- `notebooks/`, `flows/`, `config/`

## Instalar (no Terminal do Mac)
```bash
cd ~/Documents/GitHub/sector-risk-model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Cockpit interativo
```bash
cd ~/Documents/GitHub/sector-risk-model
.venv/bin/streamlit run app/cockpit/app.py
```

## Repositórios de referência
| Pasta | Onde começar |
|---|---|
| reference/Riskfolio-Lib | `examples/` — tutoriais em notebook (risk parity, CVaR) |
| reference/skfolio | `examples/` — mean_risk, risk_budgeting, clustering |
| reference/arch | `examples/` — modelos de volatilidade (GARCH) |
| reference/quantstats | `README` — métricas e tear sheets |
| reference/pandera | `docs/` — esquemas de validação |
| reference/yfinance | `doc/` — download e cache |
| reference/awesome-quant | `README.md` — mapa de bibliotecas quant |
| reference/obsidian_dataview_example_vault | abrir como vault no Obsidian para ver consultas Dataview |
