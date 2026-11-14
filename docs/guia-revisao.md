# Guia de revisão

Para quem vai rever o projeto (uma pessoa ou outra IA): o que é, onde está cada peça, o que verificar e o que já se sabe
que está fraco. Repositório: <https://github.com/fernandoxb/sector-risk-model> · cockpit: <https://sector-risk-model.streamlit.app>
· método completo: [`docs/nota-metodologica.md`](nota-metodologica.md) (português, 19 secções) · resumo em inglês:
[`docs/summary-en.md`](summary-en.md).

## 1. Em 30 segundos

Sistema de risco de mercado para os 11 ETFs setoriais SPDR e as suas 110 maiores empresas. Prevê VaR e ES (4 métodos,
modelo oficial = GARCH + cauda de Pareto, "EVT condicional"), testa as previsões fora da amostra desde 2008, mede quem
cai com quem (cópulas), liga fatores macro → setores → empresas e simula choques por Monte Carlo. Um pipeline corre
todos os dias úteis às 22:30 (Lisboa) e publica um snapshot semanal que o cockpit online lê.

## 2. Mapa do código

| Passo | Ficheiro | Saída | Testes | Nota |
|---|---|---|---|---|
| Recolha (Yahoo, FRED) | `src/collect/` | `data/raw/` | `test_empresas.py`, `test_sessao.py` | §10 |
| Validação (pandera) | `src/clean/validate.py` | `data/clean/` | `test_flow.py` (o flow pára) | §10 |
| Métricas de cauda | `src/models/cockpit.py`, `risk_metrics.py` | `sector_metrics.csv` | `test_portfolio.py` | §1 |
| Backtest VaR/ES (setores) | `src/models/backtest.py` | `var_backtest*.csv`, `official_risk.csv` | `test_backtest.py` | §6, §15 |
| Backtest das empresas | `src/models/backtest_empresas.py` | `company_backtest.csv`, `company_risk.csv` | `test_backtest_empresas.py` | §13 |
| Horizontes 10/21 dias | `src/models/horizonte.py` | `horizon_risk.csv`, `horizon_backtest.csv` | `test_horizonte.py` | §18 |
| Carteira de referência | `src/models/portfolio.py` | `risk_contribution.csv`, `stress_tests.csv` | `test_portfolio.py` | §7 |
| Scorecard 0–100 | `src/models/scorecard.py` | `scorecard.csv` | `test_scorecard.py` | §8 |
| Fatores macro | `src/models/factor_links.py` | `factor_links.csv` | `test_factor_links.py` | §9, §16 |
| Risco próprio das empresas | `src/models/decomposicao.py` | `company_factors.csv` | `test_decomposicao.py` | §16 |
| Dependência de cauda | `src/models/cauda.py` | `tail_dependence*.csv`, `tail_scenarios.csv` | `test_cauda.py` | §14, §17 |
| Motor de cenários | `src/models/cenarios.py` | `cenario_semanal.parquet`, `cenario_validacao.csv` | `test_cenarios.py` | §19 |
| Carteira do utilizador e HRP | `src/models/carteira.py` | (ao vivo, no cockpit) | `test_carteira.py` | §12 |
| Alertas, relatório semanal | `src/report/alerts.py`, `weekly.py` | Obsidian | `test_alerts.py` | §10 |
| Snapshot e publicação | `src/report/snapshot.py`, `publicar.py` | ramo `snapshot` | `test_snapshot.py`, `test_publicar.py` | §10 |
| Cockpit | `app/cockpit/app.py`, `paineis/*.py` | — | `test_cockpit_app.py`, `test_teia.py` | §11, §19 |

Todos os parâmetros estão em `config/settings.py`. As funções de cálculo são puras (sem Streamlit nem I/O).

## 3. O que verificar (por ordem de importância)

1. **Escolha do modelo sem espreitar o futuro.** `backtest.garch_filter` estima o GARCH só com dados até à véspera
   (reestimação mensal); `backtest.run(split=...)` escolhe o método em 2008–2018 (`choose_method`) e confirma em 2019–2026.
   Verificar: o teste "sem look-ahead" em `test_backtest.py` (mudar o futuro não muda o passado) e a tabela da §15.
2. **Fórmulas da EVT** (QRM 7.18 e 7.19): `backtest.gpd_quantile` e `gpd_es`; o ES na previsão é uma média por fatias da
   cauda (há um comentário `shortcut:` com o limite, para ξ alto).
3. **Testes de backtest:** Kupiec, Christoffersen, Z2 de Acerbi–Székely (p-valor por simulação sob H0), McNeil–Frey
   (bootstrap), Holm. Cada um tem um teste contra um cálculo à mão.
4. **Motor de cenários** (o ponto mais discutível): normal condicional dos fatores (`cenarios.simular`), regressão dos
   setores em 7 fatores, resíduos reamostrados, alargamento `fator_escala` e 50 conjuntos de parâmetros por bootstrap
   (`ajustar_varios`). A validação em `cenarios.episodio` estima só com as semanas antes de cada crise.
   **Resultado conhecido:** os intervalos de 90% cobrem 7 de 31 setores-crise só com fatores macro e 16 de 31 com o
   mercado real (esperado: cerca de 28). O cockpit chama-lhes "intervalos do modelo" e mostra esta cobertura.
5. **Cópulas:** ρ pelo tau de Kendall, ν por máxima verosimilhança (`cauda.fit_nu`, validado contra a tabela 5.1 do QRM);
   cópula t assimétrica com γ pelo método dos momentos (`ajustar_gama`): um parâmetro para duas médias.
6. **Regressões dos fatores:** erros HAC de Newey–West (validados contra a fórmula à mão), regressão conjunta Ridge
   (igual ao `sklearn`), aceitação por |t| ≥ 2 sem correção para testes múltiplos.
7. **HRP fora da amostra:** `carteira.hrp_fora_da_amostra` (os pesos de cada período só usam os 504 dias anteriores).
8. **Engenharia:** contrato de dados (`src/clean/validate.py`), calendário da NYSE por regras (validado contra 5 500 sessões
   do SPY), cache pela última sessão fechada, publicação semanal num ramo próprio.

## 4. Fraquezas já conhecidas (estão escritas na nota e no README)

* O motor de cenários é linear: só com fatores macro subestima as crises em cerca de 15 pontos, e os intervalos são estreitos.
* A cópula t assimétrica usa um só γ para todos os pares; alguns pares ficam abaixo do observado (XLK–XLC: 31% contra 41%).
* A correção do ES das empresas (k = 1 − Z2) é calibrada no mesmo backtest que a justifica.
* O scorecard é 70% relativo aos 11 setores; o intervalo da posição mede só a incerteza dos pesos.
* Com 3 anos de dados, cada frequência de quedas conjuntas assenta em cerca de 38 dias.
* O ES a 10 e 21 dias não tem backtest (só o VaR). A HRP fora da amostra cobre cerca de 3 anos, sem crise.
* Dados do Yahoo sem garantia de serviço; o repositório guarda só índices base 100 e resultados derivados.

## 5. Em aberto

* Metadados do repositório no GitHub (descrição, site, temas) e imagem de pré-visualização.
* Release v1.0.0.
* Scorecard: mostrar os setores por grupos (os intervalos de posição sobrepõem-se do 2.º ao 5.º lugar) em vez de posições.

## 6. Perguntas úteis para a revisão

* Há algum caminho por onde dados do dia previsto entrem na previsão (look-ahead)?
* A regra de escolha do método (`choose_method`) é razoável, ou favorece a EVT por construção?
* O alargamento dos resíduos (`fator_escala`) e o bootstrap dos parâmetros são defensáveis, ou há uma forma melhor de
  calibrar os intervalos do motor de cenários nas crises?
* O método dos momentos para γ é aceitável, ou devia ser máxima verosimilhança da cópula t assimétrica?
* Os números do README e do resumo em inglês batem com os ficheiros de `data/features/` (ou do ramo `snapshot`)?

## 7. Reproduzir

```bash
git clone https://github.com/fernandoxb/sector-risk-model && cd sector-risk-model
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                                  # sem rede
COCKPIT_SNAPSHOT=1 .venv/bin/streamlit run app/cockpit/app.py
.venv/bin/python run_pipeline.py                            # tudo de novo, com rede (10–15 minutos)
```

Os resultados mais recentes do pipeline (CSV e parquet) estão no ramo `snapshot`:
<https://github.com/fernandoxb/sector-risk-model/tree/snapshot>.
