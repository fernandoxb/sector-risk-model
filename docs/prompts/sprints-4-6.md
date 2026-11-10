# Prompt: Sprints 4–6 (motor de avaliação automática, bots, montra)

Segue o CLAUDE.md em tudo (settings.py como fonte única, funções puras, português europeu,
testes antes de cada commit, nota metodológica atualizada). Trabalha por FASES. No fim de cada
fase: corre `pytest` e `python run_pipeline.py --offline`, mostra-me os resultados pedidos e
**PARA à espera da minha aprovação** antes de fazer commit e passar à fase seguinte.

---------------------------------------------------------------------------------------------
## FASE 0 — Publicar no GitHub
1. Confirma se existe remote (`git remote -v`). Se não existir:
   `gh repo create sector-risk-model --public --source . --push`
   (se o `gh` não estiver instalado ou autenticado, diz-me o que fazer).
2. Adiciona LICENSE (MIT) e um workflow `.github/workflows/tests.yml` que corre `pytest`
   em cada push (Python 3.11, cache do pip). Os testes não podem depender de rede:
   usa dados sintéticos ou fixtures pequenas em `tests/fixtures/`.
Mostra-me: o URL do repositório e o primeiro run do workflow.

---------------------------------------------------------------------------------------------
## FASE 1 — Sprint 4a: FHS como modelo oficial + previsão de hoje
Os backtests do Sprint 2 mostram que o FHS é o único método que passa nos testes
(97,5%: Kupiec 11/11, Christoffersen 10/11, Z2 10/11; o histórico e o GARCH-t falham).
1. Em `config/settings.py`: `OFFICIAL_VAR_METHOD = "fhs"` com um comentário a citar os resultados.
2. Em `src/models/backtest.py` (reutilizando `garch_filter`/`sector_forecasts`, sem duplicar lógica):
   função pura que devolve a previsão **para o próximo dia útil** de VaR 99%, VaR 97,5% e ES 97,5%
   pelo método oficial, por setor. Teste: a previsão do último dia tem de coincidir com o que o
   backtest produziria para essa data (sem look-ahead).
3. Grava `data/features/official_risk.csv` (setor, var99, var975, es975, data_previsao).
Mostra-me: a tabela de hoje comparada com o CVaR histórico do Sprint 1.

---------------------------------------------------------------------------------------------
## FASE 2 — Sprint 4b: scorecard de risco por setor (0–100)
Novo módulo `src/models/scorecard.py`, funções puras, pesos em settings (`SCORE_WEIGHTS`).
Componentes por setor, cada um normalizado para [0, 1] (percentil entre os 11 setores, salvo indicação):
| Componente | Fonte | Notas |
|---|---|---|
| cauda | ES 97,5% oficial (FHS) de amanhã | forward-looking |
| sistemico | pct_cvar / peso em `risk_contribution.csv` | >1 = contribui mais do que pesa |
| stress | perda média nos cenários de `stress_tests.csv` | linhas com `proxy=True` pesam `STRESS_PROXY_WEIGHT = 0.5` |
| regime | `vol_pct` (percentil da vol 63d na própria história) | já é [0,1], não re-normalizar |
| momento | nº de exceções ao VaR 97,5% oficial nos últimos 60 dias | |
Mais um ajuste de **risco de modelo**, separado do score: semáforo do backtest oficial
(verde ×1,00, amarelo ×1,10, vermelho ×1,25) → `score_final`, e uma flag visível.
Saídas para o mapa (mantendo a lógica P × I de 1 a 5):
- `impacto` = quintil de (cauda, sistemico, stress) ponderados
- `probabilidade` = quintil de (regime, momento) ponderados
Grava `data/features/scorecard.csv` e acrescenta o scorecard ao `risk_map_data.json`.
Testes: normalização, pesos que somam 1, monotonia (piorar um componente nunca baixa o score),
proxy com metade do peso.
Mostra-me: o ranking dos 11 setores com os componentes e uma análise de sensibilidade
(como muda o top 3 se cada peso variar ±50%).

---------------------------------------------------------------------------------------------
## FASE 3 — Sprint 4c: ligações medidas entre fatores macro e setores
Novo módulo `src/models/factor_links.py`. Fatores em settings (`MACRO_FACTORS`), cada um com
fonte, transformação e "direção adversa":
| Nó no mapa | Série | Fonte | Variação usada |
|---|---|---|---|
| juros | DGS10 | FRED | Δ em p.p. (reutiliza `fetch_rates`) |
| inflacao | T10YIE (breakeven 10Y) | FRED | Δ em p.p. |
| spreads | BAMLH0A0HYM2 (spread high yield) | FRED | Δ em p.p. |
| vix | ^VIX | Yahoo | Δ log |
| petroleo | CL=F | Yahoo | retorno log |
| dolar | DX-Y.NYB | Yahoo | retorno log |
1. Retornos **semanais** (sexta a sexta, reduz o ruído de fusos/assincronia), janela de CVAR_YEARS.
2. Regressão de cada setor em cada fator, com erros-padrão **HAC (Newey–West)** via statsmodels.
3. Ligação fator → setor só se |t| ≥ `LINK_T_MIN = 2`. Sinal "+" se o movimento adverso do fator
   piora o setor, "−" se o melhora. Intensidade 1/2/3 por tercis de |beta padronizado|.
   Guarda beta, t, R² e n em cada ligação.
4. Ligações entre setores: as arestas da MST (`ck.mst`) como tipo "contágio" (não dirigidas).
5. Acrescenta ao `risk_map_data.json`: `links: [{from, to, sign, w, beta, t, r2, tipo}]`.
Testes: com dados sintéticos de beta conhecido, recupera o sinal e a significância.
Mostra-me: a matriz fator × setor (beta e t) e a lista de ligações aceites.

---------------------------------------------------------------------------------------------
## FASE 4 — Sprint 4d: mapa e cockpit ligados ao scorecard
Mapa (`app/risk-map/index.html`):
- Os nós de setor usam o P × I do scorecard; o painel mostra os componentes (barras horizontais),
  o ES oficial, o semáforo e a flag de risco de modelo.
- As ligações medidas aparecem automaticamente, com estilo distinto das manuais (tracejado fino
  = medida; o painel mostra beta, t e R²). O utilizador pode esconder/mostrar ligações medidas,
  e as ligações manuais continuam a funcionar.
- Os nós macro ganham "exposição": nº de setores afetados e soma ponderada dos scores a jusante.
- A cadeia de impacto usa as ligações medidas.
Cockpit: nova secção "Scorecard" (ranking, componentes, semáforo) a ler `scorecard.csv`;
nenhum cálculo duplicado — importa de `src/models/`.
Mostra-me: capturas do mapa (setor selecionado e fator "juros" selecionado) e do cockpit.

---------------------------------------------------------------------------------------------
## FASE 5 — Sprint 5: bots e alertas automáticos
1. `flows/pipeline.py` com Prefect: uma task por bot (Coletor, Inspetor, Modelador, Backtester,
   Scorecard, Repórter), retries=2 na coleta, cache por data. Se o Inspetor falhar, nada a jusante corre.
2. Agendamento no Mac com **launchd** (`flows/com.fernando.sectorrisk.plist` + instruções no README):
   todos os dias úteis às 22:30 (hora de Lisboa, depois do fecho dos EUA). Logs em `logs/`.
3. Bot Sentinela (`src/report/alerts.py`, funções puras) compara hoje com ontem e gera alertas quando:
   um setor muda de semáforo; há exceção ao VaR 97,5% oficial; o score muda de nível (P×I);
   uma ligação medida entra ou sai; o regime de juros muda.
   Escreve em `Obsidian Vault/Risco Setorial/Alertas.md` (mais recente no topo) e envia notificação
   macOS (`osascript`) só para alertas de severidade alta.
4. Relatório semanal (sextas) `Risco Setorial/Semana YYYY-Www.md`: top 3 do scorecard e variação,
   exceções da semana, ligações novas, stress mais severo, com frontmatter para o Dataview.
   Atualiza `00 Painel.md` com uma tabela Dataview dos relatórios semanais e dos alertas.
Testes: alertas com estados sintéticos de ontem/hoje.
Mostra-me: um run completo do flow e o relatório semanal gerado.

---------------------------------------------------------------------------------------------
## FASE 6 — Sprint 6: montra para o CV
1. Cockpit no **Streamlit Community Cloud**: modo snapshot quando não há rede ou o Yahoo limita
   (ficheiros pequenos em `app/cockpit/snapshot/`, versionados, atualizados pelo flow).
   Dá-me os passos para o deploy e o link final.
2. README reescrito para um recrutador (em português, com resumo em inglês no topo):
   o que é, o resultado principal (tabela do backtest: só o FHS passa), capturas em `docs/img/`,
   arquitetura (diagrama Mermaid do pipeline), como correr, metodologia, limitações honestas
   (ETFs só dos EUA, proxies em 2008 para XLRE/XLC, dados do Yahoo).
3. `docs/nota-metodologica.md` → PDF (pandoc ou weasyprint) em `docs/nota-metodologica.pdf`,
   com as fórmulas (VaR, ES, Kupiec, Christoffersen, Z2, FHS, contribuição de Euler, Ledoit–Wolf,
   HAC) e as referências.
4. Sugere-me 3 linhas para o CV e 1 parágrafo para o LinkedIn, com números reais do projeto.
Mostra-me: o README renderizado e o link do cockpit online.
