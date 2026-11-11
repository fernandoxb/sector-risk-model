# Modelo de Risco Setorial

Pipeline em Python que mede o risco dos 11 setores do S&P 500, individualmente e em conjunto.
Fases: coleta → limpeza → transformação → modelação → visualização.

## Estrutura
- `app/risk-map/` — mapa de riscos interativo (abre `index.html` no browser)
- `app/cockpit/` — cockpit Streamlit: CVaR, Downside Beta, rede MST, drill-down por ações, regime de juros e log para o Obsidian (`docs/nota-metodologica.md`)
- `src/` — código por fase (collect, clean, transform, models, report)
- `data/` — raw / clean / features (não vai para o Git)
- `reference/` — repositórios clonados para consulta (exemplos e notebooks; não vai para o Git)
- `flows/` — flow Prefect dos bots e agendamento launchd
- `notebooks/`, `config/`

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

## Bots e agendamento (Prefect + launchd)
O flow `flows/pipeline.py` corre os bots em sequência: Coletor → Inspetor → Modelador → Backtester → Carteira → Scorecard → Ligações → Repórter → Sentinela → Relatório semanal.
Se o Inspetor falhar, nada a jusante corre. O Sentinela escreve `Alertas.md` no cofre Obsidian e envia uma notificação macOS para os alertas de severidade alta. Às sextas sai `Semana AAAA-Www.md`.
```bash
.venv/bin/python -m flows.pipeline              # run completo (com download)
.venv/bin/python -m flows.pipeline --offline    # sem download
.venv/bin/python -m flows.pipeline --semanal    # força o relatório semanal
```
Agendar nos dias úteis às 22:30 (hora do Mac), com os logs em `logs/`:
```bash
sh flows/install_launchd.sh             # instala ~/Library/LaunchAgents/com.fernando.sectorrisk.plist
launchctl kickstart gui/$(id -u)/com.fernando.sectorrisk   # corre já, para testar
sh flows/install_launchd.sh --remover   # desinstala
```
O repositório está em `~/Documents`, por isso o macOS pode pedir para autorizar o acesso a essa pasta no primeiro run agendado. Se o log mostrar "Operation not permitted", dá acesso total ao disco ao `python` do `.venv` em Definições do Sistema → Privacidade e Segurança.

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
