# Modelo de Risco Setorial

[![Testes](https://github.com/fernandoxb/sector-risk-model/actions/workflows/tests.yml/badge.svg)](https://github.com/fernandoxb/sector-risk-model/actions/workflows/tests.yml)
[![Cockpit online](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sector-risk-model.streamlit.app)
[![Licença MIT](https://img.shields.io/badge/licen%C3%A7a-MIT-blue.svg)](LICENSE)

> **In English.** An end-to-end market-risk model for the 11 SPDR sector ETFs against the S&P 500. It is built as a
> portfolio project for a Master's in Risk Management. Daily VaR and Expected Shortfall come from three methods
> (historical, GJR-GARCH-t and Filtered Historical Simulation) and are **backtested out of sample from 2008 to 2026**
> with Kupiec, Christoffersen, Basel traffic lights and the Acerbi–Székely Z2 test. **FHS is the only method that passes
> coverage, independence and the ES test at the same time**, so it is the official model.
> The project also covers portfolio risk (Ledoit-Wolf covariance, Euler risk contributions, 2008/2020/2022 stress tests),
> a 0–100 risk scorecard, macro-factor → sector links estimated with HAC regressions, and an automated daily pipeline
> (Prefect + launchd) with alerts and weekly reports. There are 135 tests and CI on GitHub Actions.

Pipeline em Python que mede o risco de mercado dos 11 ETFs setoriais SPDR (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU,
XLB, XLRE, XLC) face ao SPY, setor a setor e em carteira. Corre sozinho todos os dias úteis e publica os resultados em três
sítios:
* um **cockpit** Streamlit;
* um **mapa de riscos** interativo;
* notas no **Obsidian**, com alertas e um relatório semanal.

## Resultado principal: só a FHS passa o backtest
VaR e ES a 1 dia previstos **só com dados até à véspera**, em janela móvel de 3 anos, de 2008 a 2026: 46 064 previsões
por nível, nos 11 setores. A tabela mostra em quantos setores (de 11) cada método **não é rejeitado** a 5%:

| Método | Kupiec 97,5% | Christoffersen 97,5% | Condicional 97,5% | Z2 (ES) 97,5% | Kupiec 99% | Christoffersen 99% | Condicional 99% | Z2 (ES) 99% |
|---|---|---|---|---|---|---|---|---|
| **GARCH-FHS** | **11** | **10** | **10** | **11** | **10** | **8** | **8** | **10** |
| GJR-GARCH(1,1)-t | 1 | 10 | 2 | 0 | 1 | 11 | 1 | 0 |
| Histórico | 7 | 1 | 1 | 3 | 5 | 1 | 1 | 2 |

Cada método falha por uma razão diferente:
* **Histórico:** acerta mais ou menos na frequência das exceções, mas reage tarde às crises e as exceções aparecem em grupos, por isso falha no Christoffersen.
* **GARCH com t-Student:** acompanha a volatilidade, mas a cauda paramétrica é demasiado fina, por isso falha no Kupiec e no Z2.
* **FHS:** junta a volatilidade do GARCH à cauda empírica dos resíduos, e só ela passa os três testes.

Por isso é o **modelo oficial** e publica todos os dias o VaR 99%, o VaR 97,5% e o ES 97,5% da sessão seguinte (calendário da NYSE com feriados).

![Backtest no cockpit: o histórico é rejeitado e a FHS passa](docs/img/cockpit-backtest.png)

**E empresa a empresa?** O mesmo backtest corre nas 109 maiores empresas dos 11 setores (previsões desde 2018). A FHS
continua a ser o melhor método, mas com menos folga: passa os três testes a 97,5% em **61 de 109** empresas (GARCH-t: 22;
histórico: 21), e em 40 o ES fica subestimado. Uma ação isolada tem saltos que um setor dilui. Nessas 40, o ES
mostrado é corrigido pelo fator que teria acertado no backtest (k = 1 − Z2, em média ×1,31).

## O que o projeto faz
| Bloco | Métodos | Onde |
|---|---|---|
| Risco de cauda | CVaR/ES 97,5% (FRTB), downside beta nos 20% piores dias do SPY | `src/models/cockpit.py`, `risk_metrics.py` |
| Previsão e backtest | Histórico, GJR-GARCH(1,1)-t, FHS; Kupiec, Christoffersen, Z2 de Acerbi–Székely (p-valor por simulação), semáforo de Basileia, regra por mesa do FRTB | `src/models/backtest.py` |
| Empresas e carteira própria | O mesmo backtest em 109 empresas; carteira do utilizador com contribuições de Euler, exposição por setor, stress e previsão FHS | `src/models/backtest_empresas.py`, `carteira.py` |
| Carteira | Covariância de Ledoit-Wolf, contribuições de Euler para a volatilidade e o CVaR, stress tests de 2008, COVID-19 e 2022 | `src/models/portfolio.py` |
| Scorecard 0–100 | 5 componentes normalizados por percentil, multiplicador de risco de modelo (semáforo), P × I, sensibilidade ±50% aos pesos | `src/models/scorecard.py` |
| Fatores macro | Regressões semanais setor ~ fator com erros HAC de Newey–West, para juros, inflação, spreads high yield, VIX, petróleo e dólar; contágio pela MST | `src/models/factor_links.py` |
| Bots | Prefect (uma task por bot), launchd nos dias úteis às 22:30, alertas do Sentinela e relatório semanal no Obsidian | `flows/`, `src/report/` |

| Mapa de riscos (setor selecionado) | Mapa de riscos (cadeia de impacto dos juros) |
|---|---|
| ![Mapa: XLK](docs/img/mapa-xlk.png) | ![Mapa: juros](docs/img/mapa-juros.png) |

O cockpit está organizado em separadores:
* **Resumo:** alertas do dia, setores de maior risco e previsão oficial para amanhã.
* **Mapa de risco:** a rede de contágio entre setores; clicar num setor abre as suas 10 maiores empresas e clicar numa empresa abre a ficha com dados de bolsa, risco e últimas notícias.
* **Minha carteira:** monta uma carteira com as 110 empresas, os ETFs e o SPY; mostra o risco histórico e o previsto (FHS), quem mais pesa nele, a exposição por setor no mapa de contágio e o stress. Guarda-se no link, sem conta.
* **Setores:** scorecard e evolução do score.
* **Ficha do setor:** tudo sobre um setor, incluindo o risco previsto desde 2008.
* **Análise ao vivo:** CVaR histórico, downside beta e ações.
* **Fatores macro.**
* **Modelo:** backtest.
* **Carteira-base:** a carteira de referência dos 11 setores, com contribuições para o risco, stress históricos e um cenário de stress próprio.
* **Glossário.**

| Resumo | Fatores macro |
|---|---|
| ![Cockpit: resumo](docs/img/cockpit-resumo.png) | ![Cockpit: fatores macro](docs/img/cockpit-fatores.png) |

| Mapa de risco (setor aberto) | Ficha da empresa |
|---|---|
| ![Cockpit: mapa de risco](docs/img/cockpit-mapa.png) | ![Cockpit: ficha da empresa](docs/img/cockpit-empresa.png) |

| Minha carteira |
|---|
| ![Cockpit: minha carteira](docs/img/cockpit-carteira.png) |

| Scorecard | Ficha do setor |
|---|---|
| ![Cockpit: scorecard de risco por setor](docs/img/cockpit-scorecard.png) | ![Cockpit: ficha do setor](docs/img/cockpit-ficha.png) |

**Outros números (dados até 2026-11-10):**
* **Carteira com pesos iguais:** volatilidade de 12,1% e CVaR 97,5% de 2,21% a 1 dia. A diversificação vale 1,49× na volatilidade, mas só 1,39× no CVaR, porque nas quedas os setores caem juntos.
* **Stress 2008:** a carteira perde 49,5% (SPY: 55,2%). Na COVID-19 perde mais do que o SPY (36,0% contra 33,7%), por causa do peso da energia.
* **Fatores macro:** 34 das 66 ligações fator → setor são significativas (|t| ≥ 2 com HAC). As mais fortes são o VIX → XLK (t = −14,0) e os spreads high yield → XLI (t = −11,5).
* **Scorecard:** o XLK lidera (91/100). O XLE tem um agravamento de risco de modelo, porque o seu semáforo a 99% está amarelo.

## Arquitetura
```mermaid
flowchart LR
    Y[Yahoo Finance<br/>ETFs · VIX · WTI · DXY] --> C[Coletor<br/>retries + cache]
    F[FRED<br/>juros · inflação · spreads] --> C
    C --> I{Inspetor<br/>pandera}
    I -. falha .-> X[flow pára]
    I --> M[Modelador · Backtester<br/>CVaR, GARCH, FHS<br/>Kupiec, Christoffersen, Z2]
    M --> P[Carteira · Scorecard · Ligações<br/>Ledoit-Wolf, Euler, stress<br/>score 0–100, OLS + HAC]
    P --> MAP[Mapa de riscos]
    P --> CK[Cockpit Streamlit<br/>+ snapshot p/ Cloud]
    P --> SE[Sentinela] --> OB[Obsidian<br/>alertas · relatório semanal]
    SE -. severidade alta .-> N[Notificação macOS]
```
Regras do código:
* **Parâmetros:** estão todos em `config/settings.py`, sem números mágicos noutros ficheiros.
* **Funções de cálculo:** são puras e cada uma é validada contra uma implementação independente: fórmulas à mão em numpy, `sklearn` para a Ledoit-Wolf e Newey–West à mão para o `statsmodels`.

## Como correr
```bash
git clone https://github.com/fernandoxb/sector-risk-model && cd sector-risk-model
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_pipeline.py            # coleta → validação → modelos → mapa (precisa de rede)
.venv/bin/streamlit run app/cockpit/app.py  # cockpit em http://localhost:8501
open app/risk-map/index.html                # mapa de riscos (sem servidor)
.venv/bin/python -m pytest                  # 135 testes, sem rede
```
* **Chave do FRED:** é opcional. Se definires `FRED_API_KEY` num `.env`, os dados vêm pela API; caso contrário usa-se o CSV público.
* **Sem rede:** `run_pipeline.py --offline` reutiliza os dados de `data/raw/`.

### Bots e agendamento (Prefect + launchd)
```bash
.venv/bin/python -m flows.pipeline              # run completo (com download)
.venv/bin/python -m flows.pipeline --offline    # sem download
.venv/bin/python -m flows.pipeline --semanal    # força o relatório semanal
sh flows/install_launchd.sh                     # agenda nos dias úteis às 22:30 (hora do Mac)
launchctl kickstart gui/$(id -u)/com.fernando.sectorrisk   # corre já, para testar
sh flows/install_launchd.sh --remover           # desinstala
```
* **Logs:** ficam em `logs/pipeline.log`.
* **Permissões no macOS:** como o repositório está em `~/Documents`, o macOS pode pedir autorização no primeiro run agendado. Se o log mostrar "Operation not permitted", dá acesso total ao disco ao `python` do `.venv` em Definições do Sistema → Privacidade e Segurança.
* **Obsidian:** o Sentinela escreve em `Risco Setorial/` do cofre (`OBSIDIAN_VAULT`). O `00 Painel.md` precisa do plugin Dataview.

### Cockpit no Streamlit Community Cloud
O cockpit tenta sempre os dados ao vivo. Se o Yahoo não responder (ou com `COCKPIT_SNAPSHOT=1`), lê o **snapshot versionado** em `app/cockpit/snapshot/`, com cerca de 2,7 MB: preços de 3 anos, juros e resultados do pipeline. O flow atualiza o snapshot em cada run e, na última sessão de cada semana, o bot Publicador faz commit e push só dessa pasta (`--publicar` força a publicação noutro dia). O Streamlit Cloud atualiza a app sozinho a cada push.

Para o publicar:
1. Em [share.streamlit.io](https://share.streamlit.io), entra com o GitHub e escolhe **Create app** → *Deploy a public app from GitHub*.
2. Preenche: repositório `fernandoxb/sector-risk-model`, branch `main`, ficheiro `app/cockpit/app.py`. Em *App URL* escreve `sector-risk-model`.
3. Em *Advanced settings*: Python **3.11**. Em *Secrets* escreve `COCKPIT_SNAPSHOT = "1"`.
4. Clica em **Deploy**. As dependências vêm de `app/cockpit/requirements.txt`, que é mais leve do que o `requirements.txt` da raiz.

## Metodologia
O documento completo, com as fórmulas de todos os testes e modelos, as validações e as referências, está em
[`docs/nota-metodologica.md`](docs/nota-metodologica.md), também disponível em [PDF](docs/nota-metodologica.pdf).

Convenções:
* **VaR, CVaR e beta:** retornos simples e perdas em número positivo.
* **Volatilidade e GARCH:** retornos logarítmicos.
* **Nível do ES:** 97,5%, como no FRTB.
* **Rótulos:** dizem sempre o nível e a janela, por exemplo "CVaR 97,5% · 1 dia · 3 anos".

## Limitações
* **Universo:** são só ETFs setoriais dos EUA e as suas 10 maiores empresas; o resultado não se generaliza a outros mercados.
* **Empresas:** o modelo oficial só está validado em 61 de 109 e subestima o ES em 40. O backtest das empresas começa em 2018, por isso não inclui 2008.
* **Carteira do utilizador:** a previsão usa o modelo oficial sobre os retornos da carteira com os pesos atuais. O backtest de cada carteira tem só cerca de 2 anos de previsões, sem nenhuma crise.
* **Correção do ES:** o fator é calibrado no mesmo backtest que o justifica; corrige o que o passado mostrou.
* **Dados:** os preços vêm do Yahoo Finance (ajustados, sem garantia de serviço). Há séries do FRED com 1 dia de atraso, e o spread high yield (ICE BofA) só está disponível no FRED para os últimos 3 anos.
* **Proxies em 2008:** o XLRE e o XLC ainda não existiam, por isso no stress de 2008 usam um proxy β × SPY com o β atual, que ignora o risco específico.
* **FHS:** passa os testes, mas a 97,5% tem ligeiramente mais exceções do que as esperadas no agregado (1 267 contra 1 152, ou seja, 2,75% contra 2,5%). A 99%, o XLE está em amarelo no semáforo.
* **Ligações macro:** são regressões univariadas (não isolam efeitos conjuntos) e não há correção para testes múltiplos nas 66 regressões. Com |t| ≥ 2, é de esperar cerca de 3 falsos positivos.
* **Scorecard e P × I:** são relativos aos 11 setores (percentis e quintis). Um setor pode mudar de nível sem que o seu risco mude. Na média dos stress tests, ganhos compensam perdas (o XLE em 2022).
* **Âmbito:** é um projeto académico e não é aconselhamento de investimento.

## Estrutura
```
config/settings.py      todos os parâmetros
src/collect/            Coletor (Yahoo, FRED)
src/clean/              Inspetor (pandera) + relatório de qualidade
src/models/             métricas, backtest, carteira, scorecard, ligações macro
src/report/             mapa, snapshot, Sentinela, relatório semanal
flows/                  flow Prefect e agendamento launchd
app/risk-map/           mapa de riscos (HTML autónomo)
app/cockpit/            cockpit Streamlit + snapshot
scripts/                capturas de ecrã e PDF da nota metodológica (Chrome headless)
tests/                  135 testes (pytest), sem rede (inclui o cockpit em modo snapshot)
docs/                   nota metodológica (.md e .pdf) e capturas
```

Licença MIT.
