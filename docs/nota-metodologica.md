---
tipo: documentacao-metodologica
projeto: Modelo de Risco Setorial Interativo
versao: 4.3
motor: Python (Streamlit + NetworkX + Plotly + yfinance + FRED)
app: app/cockpit/app.py
---

# 📖 Nota Metodológica: Arquitetura do Cockpit de Risco Setorial

Esta nota descreve as bases teóricas, as métricas de cauda e a estrutura de grafos do cockpit de risco. O objetivo é monitorizar o risco sistémico e o contágio entre os 11 setores do mercado (GICS) e as respetivas ações.

Para correr o cockpit: `.venv/bin/streamlit run app/cockpit/app.py` (a partir da raiz do projeto).
Código de cálculo: `src/models/cockpit.py`.

---

## 📐 1. Métricas de risco de cauda

O modelo não assume que os retornos seguem uma distribuição normal. Concentra-se nos **eventos extremos (tail risk)** através de duas métricas principais.

### A. Conditional Value at Risk (CVaR / Expected Shortfall)
O VaR diz apenas a partir de que perda começam os piores (1−α) dias. O CVaR calcula a **perda média dentro desses dias**: com α = 0,95, é a média dos 5% piores retornos diários. É histórico e usa retornos simples na janela escolhida.
$$\text{VaR}_\alpha = -Q_{1-\alpha}(r), \qquad \text{CVaR}_\alpha = \text{ES}_\alpha = -\,\mathbb{E}\left[\,r \mid r \le Q_{1-\alpha}(r)\,\right]$$
em que $Q_{1-\alpha}$ é o quantil empírico dos retornos simples diários da janela, e as perdas vêm em número positivo.

**Padrão do projeto:** α = 0,975 (o nível do Expected Shortfall no FRTB, Basileia III) e janela de 3 anos, definidos em `config/settings.py` (`CVAR_LEVEL`, `CVAR_YEARS`). O mapa de riscos e o cockpit usam a mesma função (`src/models/cockpit.py::cvar`), por isso dão o mesmo número com os mesmos parâmetros.
* **Aplicação visual:** o tamanho de cada nó do grafo é proporcional ao CVaR, já ajustado ao regime de juros (ver secção 4).

### B. Downside Beta (beta condicional de crise)
$$\beta^- = \frac{\operatorname{cov}(r_i, r_{SPY} \mid r_{SPY} \le q_p)}{\operatorname{var}(r_{SPY} \mid r_{SPY} \le q_p)}$$
Usa apenas os dias em que o SPY ficou no percentil de pânico $p$, que é ajustável no painel. Mostra que setores e ações amplificam as perdas em correções sistémicas.

---

## 🕸️ 2. Topologia de rede e filtragem de ruído (MST)

Entre 11 setores há 55 correlações possíveis, o que torna o grafo completo ilegível. Para isolar a estrutura de contágio:

1. **Distância de correlação:** $d = \sqrt{2(1 - r)}$. Com $r = 1$, a distância é 0.
2. **Minimum Spanning Tree:** liga os 11 nós com 10 arestas, escolhidas de forma a minimizar a soma das distâncias.
3. **Posições no gráfico:** usa-se o layout de Kamada-Kawai, que trata $d$ como uma distância. Assim, os setores que caem juntos ficam fisicamente próximos.

---

## 🔬 3. Duas camadas (setor → ativo)

* **Camada macro:** os 11 ETFs SPDR (XLK, XLF, XLE, …).
* **Camada micro:** as 10 maiores posições de cada ETF, no separador Mapa de risco (secção 11). O CVaR de cada empresa é comparado com o CVaR do ETF do setor **no mesmo período**, e as ligações entre empresas vêm da MST das suas correlações. O antigo "microscópio" de 4 ações por setor foi substituído por este nível.

---

## 🏦 4. Regime de juros (Treasury 10Y via FRED)

* Série `DGS10`, lida pela `fredapi` quando `FRED_API_KEY` está definida e, caso contrário, pelo CSV público do FRED.
* **Regime:** variação do 10Y nos últimos 63 dias úteis. É *subida* se ≥ +0,25 p.p., *descida* se ≤ −0,25 p.p. e *estável* nos restantes casos.
* **Choque de juros (Análise ao vivo):** o utilizador escolhe uma variação $\Delta y$ do 10Y e o CVaR de cada setor recebe por cima a perda estimada pelo seu β medido (secção 9, regressão semanal com erros HAC):
  $$\text{CVaR}_i^{choque} = \text{CVaR}_i + \max\left(0,\ -(e^{\,b_i \Delta y} - 1)\right)$$
  Só contam as ligações significativas ($|t| \ge 2$); os outros setores não mudam, e um setor que ganha com o choque fica com o CVaR original.
* **Correção de uma incoerência (2026-11):** a versão anterior penalizava uma lista fixa de "setores de crescimento" (XLK, XLC, XLY) com um parâmetro arbitrário, o que contradizia a regressão do próprio projeto. Os dados dizem que quem sofre com a subida dos juros são o XLRE (t = −4,1), o XLU (−2,7) e o XLP (−2,4); o XLK não tem ligação significativa (t = +0,6) e o XLE até ganha (+2,8). A lista fixa e o parâmetro foram retirados.

---

## 🗂️ 5. Persistência em Markdown (Obsidian)

O botão **Exportar estado de hoje** grava `<cofre>/Risco Setorial/AAAA-MM-DD.md` com:
- frontmatter com os parâmetros e o regime de juros, para consultas Dataview;
- a tabela de setores e a lista de ligações da MST, com todos os tickers como `[[Ticker]]` para gerar backlinks.

O cofre usado por omissão é `Documents/Obsidian Vault`, na subpasta `Risco Setorial/`, onde a nota `00 Painel.md` lista todos os logs com o Dataview. Pode ser alterado no painel ou com a variável `OBSIDIAN_VAULT`.

---

## 🧪 6. Backtesting do VaR e do Expected Shortfall (Sprint 2)

Código: `src/models/backtest.py` (funções puras) · parâmetros: `config/settings.py` (`BT_*`) · testes: `tests/test_backtest.py`.
Saídas: `data/features/var_backtest.csv` (resultados) e `data/features/var_backtest_series.parquet` (séries diárias para o cockpit).

### A. Previsão do VaR sem look-ahead
O VaR e o ES a 1 dia do dia $t$ são previstos **só com dados até $t-1$**, em janela móvel de $W$ = `CVAR_YEARS` × 252 = 756 dias úteis, para os níveis $\alpha \in \{99\%, 97{,}5\%\}$. Os três métodos usam retornos simples e dão o VaR e o ES em perda positiva:

* **Histórico:** $\text{VaR}_t = -Q_{1-\alpha}(r_{t-W}, \dots, r_{t-1})$, o quantil empírico (interpolação linear, igual a `risk_metrics.hist_var`). O ES é `cockpit.cvar` na mesma janela: a média das perdas a partir do VaR.
* **GJR-GARCH(1,1)-t:** estimado em retornos logarítmicos $x_t = \ln(1 + r_t)$ (em %), com média constante:
  $$x_t = \mu + \varepsilon_t,\quad \varepsilon_t = \sigma_t z_t,\quad z_t \sim t_\nu \text{ padronizada}$$
  $$\sigma_t^2 = \omega + (\alpha_1 + \gamma\, \mathbb{1}_{\varepsilon_{t-1}<0})\,\varepsilon_{t-1}^2 + \beta\, \sigma_{t-1}^2$$
  Os parâmetros são re-estimados no **1.º dia útil de cada mês** (`BT_REFIT_FREQ`) com as $W$ observações anteriores. Entre re-estimações, os parâmetros ficam fixos e $\sigma_t$ é atualizada todos os dias pela recursão. O quantil logarítmico passa a retorno simples pela transformação exata (monótona) $r = e^x - 1$:
  $$\text{VaR}_t = -\left(e^{\mu + \sigma_t\, q_{1-\alpha}(\nu)} - 1\right)$$
  O ES é a média de $-(e^{\mu + \sigma_t z} - 1)$ na cauda $z \le q_{1-\alpha}(\nu)$ da t. Calcula-se pela regra do ponto médio, com `BT_ES_GRID` = 400 quantis igualmente espaçados em probabilidade dentro de $(0, 1-\alpha)$.
* **GARCH-FHS (Filtered Historical Simulation, Barone-Adesi et al., 1999):** usa o mesmo GARCH, com os mesmos $\mu$ e $\sigma_t$, mas troca a t pela distribuição empírica dos resíduos padronizados $\hat z_s = (x_s - \hat\mu)/\hat\sigma_s$ da janela de estimação em vigor. $\text{VaR}_t = -(e^{\mu + \sigma_t \hat q} - 1)$, em que $\hat q$ é o quantil empírico $1-\alpha$ dos $\hat z$, e o ES é a média de $-(e^{\mu + \sigma_t \hat z} - 1)$ sobre os $\hat z \le \hat q$. Assim, a volatilidade vem do GARCH e a forma da cauda vem dos dados.

Os três métodos são comparados na mesma amostra: os dias em que ambos têm previsão (a partir de 2008 nos setores com história desde 2005; XLRE e XLC começam mais tarde).

### B. Exceções
$I_t = \mathbb{1}\{-r_t > \text{VaR}_t\}$. Com $T$ dias e $x = \sum I_t$ exceções, a taxa observada é $\hat p = x/T$ e a esperada é $p = 1 - \alpha$.

### C. Teste de Kupiec (POF, cobertura incondicional)
$H_0$: $\hat p = p$.
$$LR_{pof} = -2 \ln\!\left[(1-p)^{T-x} p^{x}\right] + 2 \ln\!\left[(1-\hat p)^{T-x} \hat p^{x}\right] \sim \chi^2(1)$$

### D. Teste de independência de Christoffersen
Sendo $n_{ij}$ o número de dias no estado $i$ seguidos de um dia no estado $j$ (0 = sem exceção, 1 = exceção), $\pi_0 = \frac{n_{01}}{n_{00}+n_{01}}$, $\pi_1 = \frac{n_{11}}{n_{10}+n_{11}}$ e $\pi = \frac{n_{01}+n_{11}}{n_{00}+n_{01}+n_{10}+n_{11}}$.
$H_0$: $\pi_0 = \pi_1$ (uma exceção hoje não torna mais provável outra amanhã).
$$LR_{ind} = -2 \ln\!\left[(1-\pi)^{n_{00}+n_{10}} \pi^{n_{01}+n_{11}}\right] + 2 \ln\!\left[(1-\pi_0)^{n_{00}} \pi_0^{n_{01}} (1-\pi_1)^{n_{10}} \pi_1^{n_{11}}\right] \sim \chi^2(1)$$

### E. Teste condicional combinado
$$LR_{cc} = LR_{pof} + LR_{ind} \sim \chi^2(2)$$
Rejeita-se $H_0$ quando o p-valor é inferior a `BT_SIGNIF` = 5%. Em todos os testes, $0 \cdot \ln 0 = 0$ (casos com zero exceções ou sem exceções consecutivas).

### F. Semáforo de Basileia
Conta as exceções dos **últimos 250 dias**. As zonas vêm da binomial acumulada $F(x) = P(X \le x)$, com $X \sim \text{Bin}(250, 1-\alpha)$: **verde** se $F(x) < 95\%$, **amarelo** se $95\% \le F(x) < 99{,}99\%$, **vermelho** se $F(x) \ge 99{,}99\%$. A 99% isto dá as zonas do Comité de Basileia (1996): verde 0–4, amarelo 5–9, vermelho 10+. A 97,5% usa-se o mesmo critério (verde 0–10, amarelo 11–16, vermelho 17+), o que é uma extensão do projeto e não uma regra regulamentar.

### G. Backtest do ES: teste Z2 de Acerbi–Székely (2014)
Com $L_t = -r_t$ e $p = 1 - \alpha$:
$$Z_2 = 1 - \frac{1}{T\,p} \sum_{t=1}^{T} \frac{L_t\, I_t}{\text{ES}_t}$$
Se o ES estiver bem previsto, $\mathbb{E}[L_t I_t] = p\,\text{ES}_t$ e, portanto, $\mathbb{E}[Z_2] = 0$. $Z_2 < 0$ quer dizer que as perdas nos dias de exceção são maiores ou mais frequentes do que o modelo prevê, ou seja, o ES está subestimado. O Z2 testa o VaR e o ES ao mesmo tempo.

**P-valor por simulação** (unilateral, `BT_ES_NSIM` = 1000 cenários, semente `BT_SEED`): em cada cenário, a perda de cada dia é tirada da distribuição que o próprio método previu para esse dia:
* histórico: reamostragem uniforme dos $W$ retornos anteriores;
* GARCH-t: $\mu + \sigma_t z$, com $z$ tirado da t padronizada;
* FHS: $\mu + \sigma_t \hat z$, com $\hat z$ reamostrado dos resíduos da estimação em vigor.

O $Z_2$ de cada cenário usa os VaR e ES previstos. Depois, $p = \frac{1 + \#\{Z_2^{sim} \le Z_2^{obs}\}}{1 + N}$, pelo que o menor p-valor possível é $1/1001 \approx 0{,}001$.

### H. Backtest por mesa do FRTB
No FRTB (BCBS d457, §32.5), uma mesa de negociação perde a aprovação do modelo interno se, nos últimos 250 dias, tiver **mais de 12 exceções do VaR a 99%** ou **mais de 30 a 97,5%** (`BT_FRTB_MAX`). Aplica-se aqui a cada setor como se fosse uma mesa.

### I. Validação
* O Kupiec e o Christoffersen são comparados com o cálculo feito à mão com `math.log`, e o p-valor com $\operatorname{erfc}(\sqrt{LR/2})$.
* O VaR histórico é comparado com `risk_metrics.hist_var` na janela $[t-W, t-1]$.
* O VaR GARCH (e a própria $\sigma_t$) é comparado com a recursão GJR escrita à mão a partir dos parâmetros estimados.
* O ES da t, calculado pela grelha, é comparado com a fórmula fechada $\mathbb{E}[T \mid T \le q] = -\frac{f_\nu(q)}{p}\,\frac{\nu + q^2}{\nu - 1}$, reescalada para variância 1.
* Os resíduos da FHS são comparados com $(x - \hat\mu)/\hat\sigma$ calculados à mão, e o VaR e o ES da FHS com o quantil e a média da cauda à mão.
* O Z2 é comparado com um exemplo calculado à mão. Com perdas normais e o VaR/ES normais verdadeiros, $\bar Z_2 \approx 0$; com o desvio-padrão real 1,5× maior do que o previsto, $Z_2 < -1$ e $p < 1\%$.
* A simulação histórica só usa retornos da janela $[t-W, t-1]$.
* **Sem look-ahead:** alterar o retorno do dia $t$ e de todos os seguintes não muda nenhum VaR nem ES até ao dia $t$, nos três métodos (testado num dia de re-estimação e num dia a meio do mês).
* Reutilizar os parâmetros do mês anterior como ponto de partida do otimizador altera o VaR GARCH em no máximo 0,1% e não muda nenhuma exceção (verificado no XLK).

### J. Leitura dos resultados (dados até 2026-11-09)
* **Histórico:** chumba a independência em 10 dos 11 setores. A janela de 3 anos reage devagar e as exceções aparecem em grupos nas crises.
* **GARCH-t:** reage à volatilidade e passa quase sempre a independência, mas tem exceções a mais (1,4–1,7% a 99%) e o Z2 rejeita o ES em todos os setores. A t simétrica com o ν estimado tem a cauda esquerda demasiado leve.
* **GARCH-FHS:** junta o melhor dos dois. A volatilidade do GARCH resolve o agrupamento das exceções e a cauda empírica corrige a cobertura. O Kupiec e o Z2 só a rejeitam no XLE, o setor com a cauda mais instável (choques do petróleo). O teste condicional rejeita-a no XLE e, a 99%, também no XLI e no XLB (p = 0,048 e 0,039), por exceções agrupadas.
* Nenhum setor reprova o backtest por mesa do FRTB nos últimos 250 dias, com qualquer dos métodos.

### L. A escolha do método resiste a dados que não entraram nela?
Escolher o método e avaliá-lo com os mesmos dados enviesa a conclusão (enviesamento de seleção). Para o medir, as previsões são divididas em `BT_SPLIT` = 2019-01-01. As previsões são as mesmas do backtest; só muda a amostra em que os testes são feitos. Um setor só conta num período com pelo menos 500 previsões.

Número de setores que passam os **três** testes (Kupiec, Christoffersen e Z2), dados até 2026-11-11:

| Método | Escolha 2008–2018, 97,5% | Escolha, 99% | Confirmação 2019–2026, 97,5% | Confirmação, 99% |
|---|---|---|---|---|
| **GARCH-FHS** | **6 de 9** | **9 de 9** | **9 de 11** | **8 de 11** |
| GJR-GARCH(1,1)-t | 0 de 9 | 0 de 9 | 0 de 11 | 1 de 11 |
| Histórico | 0 de 9 | 0 de 9 | 3 de 11 | 2 de 11 |

* **Escolha:** só com 2008–2018, a regra `choose_method` (mais setores a passar os três testes a 97,5%) escolhe a FHS. No período da escolha contam 9 setores, porque o XLRE e o XLC ainda não tinham previsões suficientes.
* **Confirmação:** em 2019–2026, que a escolha nunca viu e que inclui a COVID-19 e 2022, a FHS continua claramente à frente. A conclusão não depende de ter usado os mesmos dados.
* **Leitura honesta:** no período da escolha a FHS só passa os três testes a 97,5% em 6 de 9 setores; é o melhor método, não um método perfeito.

**Testes múltiplos.** Cada método é sujeito a 88 testes (11 setores × 2 níveis × 4 testes). A 5%, esperam-se cerca de 4,4 rejeições por acaso mesmo com um modelo certo, por isso uma rejeição isolada não condena um método. A correção de Holm (1979) controla a probabilidade de haver pelo menos um falso positivo, sem assumir independência entre os testes:

| Método | Rejeições a 5% (de 88) | Depois da correção de Holm |
|---|---|---|
| GARCH-FHS | 12 | 0 |
| GJR-GARCH(1,1)-t | 62 | 23 |
| Histórico | 67 | 42 |

A FHS tem mais rejeições do que as 4,4 esperadas por acaso (os testes não são independentes: o condicional combina os outros dois), mas nenhuma sobrevive à correção. Nos outros dois métodos sobrevivem dezenas.

**Validação:** `holm` é comparada com `statsmodels.stats.multitest.multipletests`; no teste dos períodos, a linha "total" é igual à do backtest sem divisão e as exceções dos dois períodos somam as do total.

### K. Modelo oficial e previsão para o próximo dia útil (Sprint 4a)
O backtest escolhe o modelo oficial: `OFFICIAL_VAR_METHOD = "fhs"`. A 97,5%, o FHS passa o Kupiec em 11/11 setores, o Christoffersen em 10/11 e o Z2 em 10/11. O histórico só passa a independência em 1/11 e o GARCH-t passa o Kupiec em 1/11 e o Z2 em 0/11. A tabela completa está no comentário de `config/settings.py`.

Todos os dias, o pipeline grava em `data/features/official_risk.csv` o **VaR 99%, o VaR 97,5% e o ES 97,5% para o próximo dia útil** (`OFFICIAL_MEASURES`), com todos os dados disponíveis até hoje.

**Sem lógica duplicada:** a série de cada setor ganha um dia extra no fim, a sessão seguinte da NYSE (calendário com feriados, ver secção 10.A), com retorno 0. Corre-se o mesmo `sector_forecasts` do backtest: a última linha é a previsão de amanhã e as restantes são o backtest. O valor 0 é só um marcador, porque a previsão de um dia usa apenas dados até à véspera. Se esse dia for o primeiro de um mês, o GARCH é re-estimado com os dados até hoje, como no backtest.

**Validação:**
* a previsão feita com dados até $t-1$ coincide com a linha $t$ do backtest completo (erro relativo $< 10^{-12}$), num dia de re-estimação, a meio do mês e no último dia;
* acrescentar o marcador não muda nenhuma linha anterior;
* no pipeline real, `var_backtest.csv` sai idêntico, byte a byte, ao de antes da alteração.

**Leitura (previsão para 2026-11-10):** o ES 97,5% oficial do XLK é 9,5%, o dobro do CVaR histórico de 3 anos (4,65%). O FHS escala a cauda pela volatilidade atual do GARCH, e o XLK acabou de ter dias de −8% e +3,7%. Nos setores calmos (XLU, XLC, XLRE), o ES oficial fica próximo ou abaixo do histórico. Este é o objetivo de um modelo condicional: o histórico pesa da mesma forma os 3 anos, enquanto o FHS reage ao regime de hoje.

---

## 🧩 7. Visão conjunta da carteira (Sprint 3)

Código: `src/models/portfolio.py` · parâmetros: `config/settings.py` (`PORTFOLIO_WEIGHTS`, `LW_*`, `STRESS_SCENARIOS`) · testes: `tests/test_portfolio.py`.
Saídas: `data/features/risk_contribution.csv`, `stress_tests.csv` e `covariance_shrinkage.csv`.

**Carteira de referência:** pesos iguais nos 11 setores ($w_i = 1/11$), salvo indicação em contrário em `PORTFOLIO_WEIGHTS`. A carteira agrega-se em **retornos simples**, porque só aí $r_p = \sum_i w_i r_i$ é exato. Por isso, nesta secção, também a volatilidade usa retornos simples, o que é uma exceção à convenção dos setores individuais. A janela é a do resto do projeto: `CVAR_YEARS` = 3 anos.

### A. Covariância de Ledoit-Wolf (2004)
Encolhe a covariância amostral $S$ (de máxima verosimilhança, dividida por $T$) para a identidade escalada:
$$\hat\Sigma = \delta\, m\, I + (1 - \delta)\, S, \qquad m = \tfrac{1}{N}\operatorname{tr}(S)$$
Com a norma $\lVert A \rVert^2 = \operatorname{tr}(AA^\top)/N$ e os retornos centrados $x_t$:
$$d^2 = \lVert S - mI \rVert^2,\quad \bar b^2 = \frac{1}{T^2}\sum_t \lVert x_t x_t^\top - S \rVert^2,\quad b^2 = \min(\bar b^2, d^2),\quad \delta = b^2/d^2$$
Para calcular $\bar b^2$ usa-se $\sum_t \lVert x_t x_t^\top - S\rVert_F^2 = \sum_t (x_t^\top x_t)^2 - T\lVert S\rVert_F^2$.

**Avaliação fora da amostra:** a carteira de variância mínima sem restrições, $w = \Sigma^{-1}\mathbf 1 / \mathbf 1^\top \Sigma^{-1}\mathbf 1$, é re-estimada no 1.º dia útil de cada mês só com os $W$ dias anteriores, uma vez com $S$ e outra com $\hat\Sigma$. Compara-se a volatilidade realizada no mês seguinte. A amostra comum aos 11 setores começa em 2018-06, com a entrada do XLC.

### B. Contribuição para o risco (decomposição de Euler)
Uma medida homogénea de grau 1 nos pesos decompõe-se em $\rho(w) = \sum_i w_i\, \partial\rho/\partial w_i$:
* **Volatilidade** (com $\hat\Sigma$ de Ledoit-Wolf): $RC_i = w_i\,(\hat\Sigma w)_i / \sigma_p$, com $\sigma_p = \sqrt{w^\top \hat\Sigma w}$.
* **CVaR histórico:** sendo $\mathcal T$ o conjunto dos dias em que $r_p \le Q_{1-\alpha}(r_p)$, ou seja, os mesmos dias que `cockpit.cvar` usa, $RC_i = -w_i\, \overline{r_{i}}\big|_{\mathcal T}$. Como $\overline{r_p}|_{\mathcal T} = \sum_i w_i \overline{r_i}|_{\mathcal T}$, a soma das contribuições é exatamente o CVaR da carteira.
* **Rácio de diversificação:** $\sum_i w_i \rho_i / \rho_p$, a soma dos riscos isolados a dividir pelo risco da carteira. Um rácio de 1 quer dizer que não há diversificação.

### C. Stress tests históricos
Três episódios, cada um do pico ao vale do SPY ajustado: **2008** (2007-10-09 → 2009-03-09), **COVID-19** (2020-02-19 → 2020-03-23) e **subida de juros** (2022-01-03 → 2022-10-12). Para cada setor, para a carteira e para o SPY calcula-se:
* a perda acumulada $1 - \prod_t (1 + r_t)$;
* o pior dia;
* o drawdown máximo dentro do episódio.

A carteira é **buy-and-hold** com os pesos no início do episódio, pelo que a perda $L_p = \sum_i w_i L_i$ é aditiva e a contribuição de cada setor é $w_i L_i$. O pior dia da carteira é também comparado com o CVaR atual, em múltiplos.

**Proxy para setores sem história no episódio** (XLRE e XLC em 2008): $r_{i,t} = \beta_i\, r_{SPY,t}$, com o β atual de `sector_metrics.csv` (3 anos). O proxy só capta a parte sistemática e ignora o risco específico, por isso é assinalado com `proxy` e com * no cockpit.

### D. Validação
* A Ledoit-Wolf, tanto $\delta$ como $\hat\Sigma$, é comparada com `sklearn.covariance.LedoitWolf` (erro relativo < $10^{-10}$), com $T$ = 30, 63 e 400.
* A variância mínima é comparada com a solução fechada para $\Sigma$ diagonal ($w_i \propto 1/\sigma_i^2$) e com perturbações aleatórias que mantêm $\sum w = 1$, nenhuma das quais pode baixar a variância.
* As contribuições de Euler somam a volatilidade e o CVaR da carteira, e a de volatilidade é comparada com a derivada numérica (diferenças finitas centradas).
* O stress test é comparado com um exemplo de 2 ativos calculado à mão, com proxy, buy-and-hold e pior dia da carteira.
* **Sem look-ahead** no teste fora da amostra: alterar os retornos a partir de $t$ não muda nada antes de $t$.

### E. Leitura dos resultados (dados até 2026-11-09)
* **Diversificação:** com pesos iguais, a volatilidade é 12,1% e o CVaR 97,5% é 2,21% a 1 dia. O rácio é 1,49× na volatilidade e só 1,39× no CVaR: nos piores dias os setores caem juntos e a diversificação encolhe.
* **Quem pesa na cauda:** o XLK, o XLY e o XLE contribuem para o CVaR mais do que o seu peso de 9,1% (12,9%, 12,4% e 11,0%). O XLP, o XLU e o XLV contribuem bastante menos (4,7–6,8%). O XLE pesa mais no CVaR (11,0%) do que na volatilidade (7,6%), porque as suas quedas coincidem com os piores dias da carteira.
* **Stress:** em 2008 a carteira perde 49,5% (SPY: 55,2%) e o maior contribuinte é o XLF. Na COVID perde **mais** do que o SPY (36,0% vs 33,7%), porque os pesos iguais dão à energia mais peso do que o índice. Em 2022 perde menos (17,2% vs 24,5%), porque o XLE subiu 44,5%. O pior dia da COVID equivale a 5,4× o CVaR atual, o que mostra que o CVaR de 3 anos não é um cenário de stress.
* **Ledoit-Wolf:** com $T/N$ = 69 (3 anos) o encolhimento é pequeno ($\delta$ ≈ 0,04) e reduz a volatilidade fora da amostra em 2,2%. Com 63 dias ($T/N$ ≈ 6), $\delta$ ≈ 0,27 e o número de condição desce de 33 para 9, mas o ganho fora da amostra é marginal (0,2%). Com apenas 11 ativos a covariância amostral já está bem condicionada, e o benefício da Ledoit-Wolf vai crescer quando se descer ao nível das ações.

### F. Cenário de stress próprio (cockpit)
O utilizador escolhe **um** choque instantâneo, no SPY ou num fator macro, e o efeito propaga-se pelos betas já medidos (`portfolio.shock_returns`):
* **Mercado:** $r_i = \beta_i \cdot \Delta_{SPY}$, com o beta diário de 3 anos em retornos simples (`sector_metrics.csv`).
* **Fator macro $k$:** $r_i = e^{\,b_{ik} x} - 1$, com o $b_{ik}$ semanal da regressão HAC da secção 9 (retornos log do setor). Usam-se só as ligações aceites ($|t| \ge 2$); as outras contam 0. A variável é $x = \Delta$ em p.p. para juros, inflação e spreads, e $x = \ln(1 + \text{choque})$ para o VIX, o petróleo e o dólar.
* **Carteira:** $r_p = \sum_i w_i r_i$, com os pesos da secção 7. Mostra-se também em múltiplos do CVaR diário da carteira.

**Limitações.** É uma aproximação linear e não soma choques: os fatores estão correlacionados entre si e com o mercado, e somar os efeitos univariados contaria o mesmo risco duas vezes. Choques grandes extrapolam para fora da amostra em que os betas foram estimados e não captam o contágio próprio das crises. Para isso servem os cenários históricos da secção C.

---

## 🎯 8. Scorecard de risco por setor, 0–100 (Sprint 4b)

Código: `src/models/scorecard.py` · parâmetros: `config/settings.py` (`SCORE_*`, `STRESS_PROXY_WEIGHT`, `MOMENTUM_DAYS`, `MODEL_RISK_*`) · testes: `tests/test_scorecard.py`.
Saídas: `data/features/scorecard.csv`, `scorecard_sensitivity.csv` e a chave `scorecard` em `risk_map_data.json`.

### A. Componentes (maior = mais risco)
| Componente | Valor em bruto | Fonte | Peso |
|---|---|---|---|
| cauda | ES 97,5% oficial (FHS) para o próximo dia útil | `official_risk.csv` | 0,30 |
| sistemico | % do CVaR da carteira ÷ peso (>1 = contribui mais do que pesa) | `risk_contribution.csv` | 0,20 |
| stress | perda média nos 3 cenários históricos | `stress_tests.csv` | 0,20 |
| regime | percentil da vol 63d face à história do próprio setor | `sector_metrics.csv` | 0,15 |
| momento | exceções ao VaR 97,5% oficial nos últimos 60 dias | `var_backtest_series.parquet` | 0,15 |

**Normalização:** cada componente vai para $[0,1]$ pelo percentil entre os 11 setores, $c = (\text{posição} - 1)/(n-1)$, com empates na posição média. Assim, o setor com o valor mais baixo vale 0 e o mais alto vale 1. O *regime* já é um percentil em $[0,1]$ e não é re-normalizado.

**Stress com proxy:** nos cenários em que o setor usa o proxy β × SPY (XLRE e XLC em 2008), a perda pesa `STRESS_PROXY_WEIGHT` = 0,5:
$$\text{stress}_i = \frac{\sum_s \omega_s L_{i,s}}{\sum_s \omega_s},\qquad \omega_s = 0{,}5 \text{ se proxy, } 1 \text{ caso contrário}$$

### B. Score, risco de modelo e P × I
$$\text{score}_i = 100 \sum_k w_k\, c_{i,k},\qquad \sum_k w_k = 1$$
**Risco de modelo:** o score é multiplicado pelo semáforo de Basileia do método oficial a 99%, o nível em que as zonas estão definidas: verde ×1,00, amarelo ×1,10, vermelho ×1,25. O resultado é o `score_final`, limitado a 100, e a flag `risco_modelo` fica ativa quando o semáforo não é verde. É um ajuste separado: o score mede o risco do setor; o multiplicador mede a confiança que podemos ter na medição.

**Mapa:** o **impacto** é o quintil (1–5, a mesma regra `quintile_score` do Sprint 1) da média ponderada de cauda, sistémico e stress. A **probabilidade** é o quintil da média ponderada de regime e momento, com os pesos de cada grupo re-normalizados.

### C. Validação
* A normalização é testada nos extremos (o menor dá 0, o maior dá 1), nos empates e com o regime intacto.
* Os pesos têm de somar 1, e os grupos de impacto e de probabilidade são uma partição dos componentes. O score é também comparado com um cálculo à mão.
* **Monotonia:** piorar qualquer componente de qualquer setor nunca baixa o score desse setor, nem o impacto ou a probabilidade do grupo afetado. Um semáforo pior também nunca baixa o `score_final`.
* O proxy pesa metade, verificado à mão. O momento só conta a janela pedida.

### D. Sensibilidade e leitura (2026-11-10)
* **Topo:** o XLK é o 1.º com folga (score 91). Tem o maior ES previsto (9,5%), a maior contribuição para o CVaR por unidade de peso (1,42×) e a volatilidade no percentil 88%. Continua em 1.º em todas as variações de ±50% em qualquer peso.
* **Logo abaixo:** XLE, XLY, XLI e XLF ficam num grupo apertado (score final entre 67 e 72). Por isso o 2.º e o 3.º lugares mudam com os pesos: em 5 das 10 variações o top 3 muda de composição.
* **XLE:** é o único setor com o semáforo amarelo do modelo oficial. Sobe de 4.º para 2.º só por causa do ajuste de risco de modelo.
* **Limitação do stress:** a *perda média* deixa os ganhos compensar as perdas. O XLE subiu 44,5% em 2022, por isso o seu componente de stress é o mais baixo (0), apesar de ter perdido 48% e 56% nos outros dois cenários. Uma alternativa a considerar é usar só as perdas, ou o pior cenário.

---

## 🔗 9. Ligações medidas entre fatores macro e setores (Sprint 4c)

Código: `src/models/factor_links.py` e `src/collect/download_factors.py` · parâmetros: `MACRO_FACTORS`, `LINK_*` · testes: `tests/test_factor_links.py`.
Saídas: `data/features/factor_links.csv` (as 66 regressões) e a chave `links` em `risk_map_data.json`.

### A. Fatores
| Nó no mapa | Série | Fonte | Variação semanal | Direção adversa |
|---|---|---|---|---|
| juros | DGS10 (Treasury 10Y) | FRED | Δ em p.p. | subir |
| inflacao | T10YIE (breakeven 10Y) | FRED | Δ em p.p. | subir |
| spreads | BAMLH0A0HYM2 (spread high yield) | FRED | Δ em p.p. | subir |
| vix | ^VIX | Yahoo | Δ log | subir |
| petroleo | CL=F (WTI, futuro) | Yahoo | Δ log | subir |
| dolar | DX-Y.NYB (DXY) | Yahoo | Δ log | subir |

**Retornos semanais** (sexta a sexta, com o último valor de cada semana). Reduzem o ruído da assincronia entre fontes: o FRED publica com 1 dia de atraso e os futuros fecham a horas diferentes das ações. A semana em curso, ainda incompleta, fica de fora. A janela é de `CVAR_YEARS` = 3 anos (156–157 semanas). Os setores usam retornos log semanais.

### B. Regressão e erros-padrão HAC
Para cada setor $i$ e fator $k$: $r_{i,t} = a + b\,\Delta f_{k,t} + \varepsilon_t$ por OLS. Os retornos semanais podem ter autocorrelação e heterocedasticidade, por isso os erros-padrão são de **Newey–West (HAC)**:
$$\widehat{V}(\hat b) = (X^\top X)^{-1}\, \hat S\, (X^\top X)^{-1},\qquad \hat S = \sum_t \hat u_t \hat u_t^\top + \sum_{l=1}^{L}\left(1 - \frac{l}{L+1}\right)\sum_t \left(\hat u_t \hat u_{t-l}^\top + \hat u_{t-l} \hat u_t^\top\right)$$
com $\hat u_t = x_t \hat\varepsilon_t$, o núcleo de Bartlett e $L = \lfloor 4 (T/100)^{2/9} \rfloor$ = 4 lags para $T$ = 157.

### C. Aceitação, sinal e intensidade
* **Aceite** se $|t| \ge$ `LINK_T_MIN` = 2.
* **Sinal** (a mesma convenção das ligações manuais do mapa): **+** se o movimento adverso do fator piora o setor ($\text{adverso} \cdot b < 0$); **−** se o melhora.
* **Intensidade** 1/2/3 pelos tercis de $|b_{pad}|$ entre as ligações aceites, com $b_{pad} = b\, \sigma_{\Delta f}/\sigma_r$, que numa regressão simples é a correlação.
* Guardam-se $b$, $t$, $R^2$ e $n$ em cada ligação.
* **Contágio entre setores:** as 10 arestas da MST de `cockpit.mst`, sobre os retornos diários de 3 anos, com o tipo "contágio" (sem direção). A intensidade vem dos tercis da correlação.

### D. Validação
* Com dados sintéticos de beta conhecido (+0,20, −0,30 e 0), a regressão recupera o beta (±0,03) e a significância ($|t| > 2$ quando há relação, $|t| < 2$ quando não há). O sinal segue a direção adversa e inverte quando a direção adversa se inverte.
* O erro-padrão HAC é comparado com a fórmula de Newey–West escrita à mão em numpy, com $L$ = 0, 2 e 4 (erro relativo < $10^{-10}$). Nota: o `statsmodels` usa o Newey–West clássico, sem a correção $n/(n-k)$.
* Também são testados o corte da semana incompleta, as transformações (Δ e Δ log, com níveis ≤ 0 → sem valor), o último valor de cada semana e a MST com $N-1$ arestas.

### E. Resultados (semanas 2023-11-10 → 2026-11-06)
São aceites **33 das 66** ligações. Por fator:

| Fator | Ligações aceites | Leitura |
|---|---|---|
| vix | 10 | O medo afeta todos os setores menos o XLE; mais o XLK ($R^2$ 0,53) e o XLY, menos o XLP e o XLU (defensivos). |
| spreads | 9 | O crédito a alargar pesa mais no XLI, no XLF e no XLK ($t$ ≈ −11). O XLP e o XLU não são afetados. |
| dolar | 5 | O dólar forte pesa no XLB, no XLP, no XLV, no XLC e no XLRE (multinacionais e matérias-primas). |
| juros | 4 | A subida de juros pesa no XLRE ($t$ = −4,1), no XLU e no XLP (setores "obrigacionistas") e ajuda o XLE. |
| inflacao | 3 | O breakeven a subir ajuda o XLE, o XLF e o XLI (reflação), por isso o sinal é −. |
| petroleo | 2 | O petróleo a subir ajuda o XLE ($R^2$ 0,37) e pesa no XLY (consumo). |

**Comparação com as ligações manuais do mapa:**
* confirmadas: juros → XLRE (+), juros → XLU (+), petróleo → XLE (−), petróleo → XLY (+), spreads → XLF (+), VIX → XLK (+);
* **não confirmadas** nestes 3 anos: juros → XLK (+), com $t$ = 0,56, e juros → XLF (−), com $t$ = 0,13.

**Limitações:**
* **Testes múltiplos:** com 66 testes a $|t| \ge 2$ (cerca de 5%), esperam-se ~3 ligações falsas mesmo sem relação nenhuma. As ligações com $|t|$ perto de 2 (dólar → XLRE, inflação → XLI) devem ser lidas com cautela.
* As regressões são **univariadas**: o VIX e os spreads estão muito correlacionados entre si, e cada beta capta também o efeito dos outros fatores.
* O CSV público do FRED só tem **os últimos 3 anos** do spread high yield (licença do ICE). Chega para `CVAR_YEARS` = 3, mas não para janelas mais longas.

### F. No mapa de riscos (Sprint 4d)
* **Nós de setor:** P × I vem do scorecard (secção 8). O painel mostra o score final, o semáforo, a flag de risco de modelo, o ES 97,5% oficial e os 5 componentes em barras.
* **Ligações medidas:** são desenhadas automaticamente entre os nós presentes no mapa: as ligações fator → setor a tracejado fino e as de contágio (MST) a pontilhado, sem seta. Ao clicar numa, o painel mostra β, t (HAC), R² e n. Podem ser escondidas com **Ligações medidas**. As ligações manuais não mudam e continuam guardadas no navegador.
* **Exposição de um nó macro:** o número de setores que o fator agrava (+) e que atenua (−), e a soma ponderada dos scores a jusante, $\sum_{\text{ligações }+} (w/3)\cdot \text{score\_final}_{setor}$. Só contam as ligações "+", porque são as que transmitem o choque adverso.
* **Cadeia de impacto:** percorre as ligações manuais e as medidas fator → setor, a jusante do nó selecionado. Também destaca as origens diretas desse nó e os seus vizinhos de contágio. As arestas da MST não entram na cadeia: não têm direção e, como ligam todos os setores, acenderiam o mapa inteiro.
* `index.html#sel=xlk` abre o mapa com um risco já selecionado.

---

## 🤖 10. Bots, alertas e relatório semanal (Sprint 5)

O pipeline corre sozinho nos dias úteis às 22:30 (hora de Lisboa, depois do fecho dos EUA): `launchd` → `flows/pipeline.py` (Prefect).

### A. Flow
* **Calendário da NYSE (`src/collect/sessao.py`):** os feriados são calculados pelas regras da bolsa, sem dependências:
  * data fixa (Ano Novo, Juneteenth desde 2022, Independência, Natal), passando para sexta se calharem ao sábado e para segunda se calharem ao domingo; o Ano Novo ao sábado não fecha a sexta;
  * data móvel (Martin Luther King, Presidents' Day, Sexta-feira Santa pela Páscoa, Memorial Day, Labor Day, Thanksgiving);
  * mais os fechos excecionais de `NYSE_SPECIAL_CLOSURES` (furacão Sandy, lutos nacionais).
  * **Validação:** de 2005-01-03 a 2026-11-11, as sessões do calendário coincidem dia a dia com as 5 500 sessões em que o SPY negociou. O teste repete a comparação com os 3 anos do snapshot no CI.
  * **Fechos antecipados (13:00 em NY):** 3 de julho, o dia a seguir ao Thanksgiving e a véspera de Natal, quando são dias de sessão. Nesses dias a sessão conta como fechada às 14:00 de NY. A regra reproduz os fechos antecipados de 2023 a 2025 publicados pela bolsa, mas não se consegue validar com preços diários, e a do 3 de julho teve exceções históricas.
  * **No cockpit:** a barra lateral mostra a última sessão fechada, a próxima sessão, os 3 próximos feriados e o próximo fecho antecipado, e avisa se os resultados do pipeline estiverem atrás da última sessão.
  * **Onde se usa:** na sessão esperada (cache e alerta de dados atrasados), na data da previsão oficial (secção 6.K) e no dia do relatório semanal e da publicação.
* **Uma task por bot:** Coletor → Inspetor → Modelador → Backtester → Backtester de empresas → Carteira → Scorecard → Ligações → Repórter → Sentinela → Relatório → Snapshot → Publicador.
* **Publicador (`src/report/publicar.py`):** na última sessão de cada semana (normalmente sexta), depois do run, faz commit e push só de `app/cockpit/snapshot/`, para o cockpit do Streamlit Cloud acompanhar os dados. Só atua no ramo `main` e só se o snapshot mudou. Usa `git commit --only`, pelo que nunca leva outras alterações, mesmo que já estejam no índice. Se o push falhar, o commit fica local, o erro vai para o log e chega uma notificação; o resto do run não é afetado.
* **Snapshot (Sprint 6):** copia para `app/cockpit/snapshot/` (versionado, cerca de 2,7 MB) os preços dos ETFs e das empresas do mapa de risco (5 anos, desde a secção 13), o Treasury 10Y e os resultados que o cockpit lê. Assim o cockpit corre sem rede, por exemplo no Streamlit Community Cloud. As séries do backtest são gravadas em float32, o que dá um erro relativo de cerca de $10^{-7}$, sem efeito nas exceções mostradas.
* **Coletor:** 2 novas tentativas, 5 minutos entre elas, e cache pela **última sessão de Nova Iorque já fechada** (17:00 hora de NY, 1 h depois do fecho, `src/collect/sessao.py`). Dois runs que esperam a mesma sessão descarregam os dados uma só vez, e a cache só é usada se os preços já chegarem a essa sessão. Se o download correr mas os preços vierem atrasados, o Coletor repete-o até 3 vezes, com 10 minutos entre elas; se continuarem atrasados, o flow segue e o Sentinela alerta. A chave não pode ser a data do calendário: um run de madrugada em Lisboa ainda só tem a sessão da véspera e bloquearia o run das 22:30 desse dia (aconteceu a 2026-11-11).
* **Paragem em caso de falha:** as tasks correm em sequência. Se o contrato do Inspetor falhar, a exceção pára o flow e nada a jusante corre, por isso nunca se publica um mapa ou um alerta calculado com dados inválidos.
* **Fatores macro no Inspetor:** passam a ter contrato pandera: uma coluna por fator, valores finitos, datas únicas e ordenadas. Os níveis <= 0 em séries com variação logarítmica são sinalizados sem ser apagados (o WTI de −37,63 a 2020-04-20 fica de fora pelo `logdiff`). A data do último valor de cada fator vai para `quality_report.md`, porque as séries do FRED chegam com 1 dia de atraso.

### B. Bot Sentinela (`src/report/alerts.py`)
Em cada run grava um estado compacto do dia em `data/state/estado_<data>.json`: score, semáforo, P × I, perda e VaR/ES oficiais, ligações aceites e regime de juros. Depois compara-o com o último estado de uma data anterior.

| Alerta | Regra | Severidade |
| --- | --- | --- |
| Semáforo | muda o semáforo de Basileia (VaR 99% FHS, 250 dias) | piora para vermelho: alta; piora para amarelo: média; melhoria: baixa |
| Exceção | perda do dia > VaR 97,5% FHS previsto com dados até t−1 | média; alta se também > ES 97,5% |
| Nível P × I | muda o nível de P × I: baixo ≤ 4, moderado ≤ 9, elevado ≤ 16, crítico ≤ 25 | sobe para crítico: alta; outra subida: média; descida: baixa |
| Ligação medida | uma ligação fator → setor passa a ser aceite (\|t\| ≥ 2) ou deixa de o ser | entra: média; sai: baixa |
| Regime de juros | muda o regime do Treasury 10Y (variação em 63 dias úteis) | para subida: alta; outra mudança: média |
| Dados atrasados | os preços acabam antes da última sessão de NY já fechada (falha na coleta; os feriados já estão no calendário) | alta |

* **Exceções:** precisam só do próprio dia, por isso também saem no primeiro run.
* **Alertas.md:** o Sentinela escreve `Risco Setorial/Alertas.md`, com uma secção por data e a mais recente no topo. Cada alerta tem campos inline do Dataview (`[severidade:: alta]`). Repetir o run do mesmo dia reescreve a secção, sem duplicar.
* **Notificação:** só os alertas de severidade alta geram uma notificação macOS (`osascript`).
* **Limitação:** o nível P × I usa quintis entre os 11 setores, por isso um setor pode mudar de nível sem que o seu score se mexa, apenas porque outro mudou de posição. Ex.: XLI a 2026-11-10, P×I 16 → 20 com score 69,3 → 69,2.

### C. Relatório semanal (`src/report/weekly.py`)
* **Quando:** sai na última sessão da NYSE de cada semana: normalmente sexta, quinta se a sexta for feriado, como a Sexta-feira Santa (ou com `--semanal`), e é gravado em `Risco Setorial/Semana AAAA-Www.md`.
* **Conteúdo:**
  * top 3 do scorecard, com a variação de score e de posição;
  * exceções ao VaR 97,5% oficial na semana, face às esperadas (2,5% × dias × 11);
  * ligações que entraram ou saíram;
  * cenário de stress mais severo para a carteira e o setor mais atingido nesse cenário.
* **Referência:** o último estado da semana anterior; na primeira semana com estados, o primeiro estado da própria semana.
* **Painel:** o frontmatter (`tipo: relatorio-semanal`, `semana`, `top3`, `excecoes`, …) alimenta `00 Painel.md`, que agrega relatórios semanais, alertas recentes e logs diários do cockpit.

### D. Validação
`tests/test_alerts.py` (22 testes) usa estados sintéticos de ontem e de hoje e cobre:
* cada tipo de alerta e as suas severidades, em ambos os sentidos;
* que não há alertas quando nada muda e que as exceções saem mesmo sem estado anterior;
* a idempotência de `Alertas.md` e o corte das secções antigas;
* a notificação só para severidade alta;
* a escolha do estado de referência;
* a semana ISO, as exceções da semana e o cenário de stress mais severo.

---

## 🕸️ 11. Mapa de risco: setor → empresas → ficha da empresa

O mapa de risco é um só e vive no cockpit (separador **Mapa de risco**). Parte da espinha de contágio (MST das correlações entre setores, secção 2) e desce dois níveis.

### A. Empresas de cada setor
* **Universo:** as `HOLDINGS_TOP` = 10 maiores posições de cada ETF, com o peso no ETF (Yahoo Finance, dados do fundo), num total de 110 empresas. Somam entre 37% (XLI) e 78% (XLE) do setor. Ficam em `data/features/holdings.csv` e são atualizadas pelo Coletor em cada run.
* **Falhas:** se o Yahoo falhar num setor, mantêm-se as posições do ficheiro anterior; sem ficheiro, usam-se as 4 ações de `SECTOR_PROFILE` com pesos iguais.

### B. Setor aberto
* **Ficha do setor:** com um setor aberto e nenhuma empresa escolhida, o painel mostra o score e os seus componentes face à mediana, os fatores macro que o agravam ou atenuam, os vizinhos de contágio, o retorno nas crises e, numa secção que abre, o risco previsto desde 2008 e o backtest do modelo nesse setor.
* **Desenho:** o setor vai para o centro, as suas empresas ficam num anel interior e os outros setores num anel exterior, pela ordem angular que tinham no mapa. Só se desenham as ligações de contágio do setor aberto.
* **Empresa:** o tamanho do círculo é o peso no ETF; a cor é o rácio entre o CVaR 97,5% da empresa e o do ETF (verde ≤ 1×, vermelho ≥ 2,5×), na janela escolhida na barra lateral.
* **Ligações entre empresas:** ao escolher uma empresa, mostram-se as suas arestas na MST das correlações entre as empresas do setor, ou seja, as que mais se movem com ela.
* **Métricas:** CVaR histórico, downside beta, beta, correlação com o ETF e volatilidade anual, com as mesmas funções dos setores (`cockpit.cvar`, `cockpit.tail_table`).

### C. Ficha da empresa
* **Bolsa:** preço e variação do dia, capitalização, PER, intervalo de 52 semanas, dividendo e recomendação dos analistas, do Yahoo Finance.
* **Notícias:** as `COMPANY_NEWS` = 6 mais recentes, com fonte, hora (UTC) e ligação.
* **Origem:** ao vivo, com cache de 15 minutos. Se o Yahoo falhar, ou no modo snapshot, usa-se `app/cockpit/snapshot/companies.json`, gravado no último run; a ficha indica sempre a origem.
* **Ligação direta:** `?setor=XLK&empresa=NVDA` abre o mapa já nesse setor e nessa empresa.

### D. Limitações
* As medidas da janela (CVaR histórico, betas, correlações) são descritivas. A previsão e o backtest por empresa estão na secção 13: o modelo oficial está validado em 61 de 109 empresas.
* Com 10 empresas por setor, o mapa não cobre o setor todo, e a MST entre empresas só vê essas 10.
* As posições e os pesos são os atuais: aplicá-los a 3 anos de preços ignora mudanças na composição dos ETFs.
* A ficha e as notícias dependem do Yahoo Finance, sem garantia de serviço; os textos vêm em inglês.

---

## 💼 12. Carteira do utilizador

Separador **Minha carteira** do cockpit; cálculo em `src/models/carteira.py` (funções puras) · testes: `tests/test_carteira.py`.

### A. Carteira
* **Ativos:** as 110 empresas do mapa de risco, os 11 ETFs setoriais e o SPY, até `USER_PORTFOLIO_MAX` = 25 posições, só com pesos positivos (sem vendas a descoberto). Os pesos são ajustados em proporção para somar 100%.
* **Onde se guarda:** no endereço da página (`?carteira=NVDA:40,JPM:30,XLU:20,SPY:10`), sem contas de utilizador. Entradas inválidas ou fora do universo são ignoradas.

### B. Medidas
* **Risco da carteira:** volatilidade anual (covariância Ledoit-Wolf) e CVaR 97,5% histórico a 1 dia, na janela escolhida, com as mesmas funções da carteira de referência (secção 7).
* **Contribuições de Euler:** a parte de cada posição no CVaR, $RC_i = -w_i\,\overline{r_i}|_{\mathcal T}$, soma exatamente o CVaR da carteira. O rácio "risco / capital" é $(RC_i / \text{CVaR}_p) / w_i$: acima de 1, a posição pesa mais no risco do que no dinheiro investido.
* **Beta e downside beta** da carteira face ao SPY; o beta é a média ponderada dos betas das posições.
* **Exposição por setor:** cada empresa conta no seu ETF (no de maior peso, se estiver em dois), cada ETF no próprio setor e o SPY em "Mercado".
* **Score de risco setorial:** média dos scores dos setores (secção 8) ponderada pela exposição; só conta a parte investida em setores.

### C. Stress
* **Cenários históricos:** $L_p \approx \sum_s e_s L_s$, em que $e_s$ é a exposição ao setor $s$ e $L_s$ a perda desse setor no episódio (secção 7.C). Cada posição leva a perda do seu setor.
* **Choque no mercado:** $r_p = \sum_i w_i \beta_i \Delta_{SPY}$ (secção 7.F), com os betas da janela escolhida.

### D. Limitações
* As contribuições, a exposição e o stress são **históricos e descritivos**. O VaR e o ES previstos da carteira vêm do modelo oficial, com um backtest curto da própria carteira (secção 13.C).
* O stress histórico ignora o risco específico de cada empresa, pelo que **subestima** carteiras concentradas em poucas ações.
* O universo são as 10 maiores posições de cada ETF; não há obrigações, ações fora dos EUA nem outras classes de ativos.
* Pesos fixos: não há histórico de transações nem re-equilíbrio.

---

## 🏢 13. Modelo de risco por empresa e previsão da carteira

Código: `src/models/backtest_empresas.py` · testes: `tests/test_backtest_empresas.py` · saídas: `company_backtest.csv` e `company_risk.csv`.

### A. O mesmo backtest, empresa a empresa
* **Sem lógica nova:** corre-se `backtest.run` (secção 6) sobre as empresas do mapa: histórico, GJR-GARCH(1,1)-t e FHS, com previsões feitas só com dados até à véspera e os mesmos testes.
* **Dados:** preços desde `COMPANY_START` = 2015, validados pelo Inspetor. Com a janela de 756 dias, as previsões começam em 2018 (mediana de 2 226 dias por empresa) e apanham a COVID-19 e 2022, mas não 2008.
* **Sem modelo:** empresas com menos de `COMPANY_MIN_DAYS` = 816 dias de preços ficam só com o risco histórico (hoje, a GEV).

### B. Resultados (109 empresas, dados até 2026-11-11)
Número de empresas em que o teste **não rejeita** o modelo a 5%:

| Método | Kupiec 97,5% | Christoffersen 97,5% | Condicional 97,5% | Z2 (ES) 97,5% | Kupiec 99% | Christoffersen 99% | Condicional 99% | Z2 (ES) 99% |
|---|---|---|---|---|---|---|---|---|
| **GARCH-FHS** | **86** | **96** | **91** | **69** | **98** | **101** | **97** | **88** |
| GJR-GARCH(1,1)-t | 43 | 94 | 52 | 24 | 59 | 104 | 69 | 32 |
| Histórico | 79 | 39 | 32 | 64 | 80 | 46 | 44 | 63 |

* **A FHS continua a ser o melhor método,** mas com menos folga do que nos setores: passa os três testes a 97,5% (Kupiec, Christoffersen e Z2) em **61 de 109** empresas, contra 22 do GARCH-t e 21 do histórico.
* **Onde falha:** em 40 empresas o Z2 rejeita com $Z_2 < 0$, ou seja, o **ES fica subestimado**. Uma ação isolada tem saltos (resultados, notícias) que nem o GARCH nem os resíduos passados antecipam e que um setor dilui. A taxa média de exceções a 97,5% é 2,98%, acima dos 2,5%.
* **Semáforo a 99%:** 87 empresas em verde e 22 em amarelo; nenhuma em vermelho.
* **Consequência:** cada empresa leva duas marcas em `company_risk.csv`: `validado` (passa os três testes a 97,5%) e `es_subestimado`. O cockpit mostra-as na ficha e na tabela do setor.

**Correção do ES subestimado.** O Z2 é linear em $1/\text{ES}$: multiplicar o ES previsto por $k$ dá $Z_2' = 1 - (1 - Z_2)/k$. Por isso $k = 1 - Z_2$ é exatamente o fator que teria posto o Z2 a zero no backtest. Quando o Z2 rejeita com $Z_2 < 0$:
$$\text{ES}^{aj} = k \cdot \text{ES}, \qquad k = \min(1 - Z_2,\ \texttt{ES\_ADJ\_MAX} = 1{,}5)$$
* Só agrava: sem rejeição, ou com $Z_2 \ge 0$, $k = 1$. Nas 40 empresas corrigidas, $k$ é em média 1,31.
* O VaR não é corrigido: a correção trata o tamanho das perdas, não a sua frequência.
* **Limitação:** $k$ é calibrado no mesmo backtest que o justifica (dentro da amostra). Corrige o que o passado mostrou, não garante o futuro.

**Cadência.** O backtest completo das empresas demora cerca de 3 minutos e 20 segundos, por isso só corre na última sessão de cada semana, ou quando as empresas mudam (`COMPANY_FULL_WEEKLY`). Nos outros dias refaz-se apenas a previsão para a sessão seguinte, com uma ou duas estimações do GARCH por empresa (6 segundos no total). O semáforo e as marcas ficam então com até uma semana de atraso.

### C. Previsão da carteira do utilizador
* **Método:** aplica-se o modelo oficial aos retornos da própria carteira com os pesos atuais, $r_{p,t} = \sum_i w_i r_{i,t}$ em cada dia passado (FHS de carteira com pesos constantes). O GARCH é estimado sobre $r_p$ com a mesma janela e as mesmas regras (`backtest.next_day_forecast`).
* **Porquê assim:** o VaR e o ES não são aditivos; somar as previsões das posições ignoraria a diversificação. Modelar $r_p$ diretamente capta-a sem estimar correlações dinâmicas.
* **Dados:** 5 anos de preços (`USER_FORECAST_YEARS`), nos dias em que todas as posições têm preço. Sem 756 dias + 1 mês em comum, não há previsão.
* **Backtest da própria carteira:** o cockpit corre `backtest.run` sobre $r_p$ e mostra as exceções, os testes de Kupiec, Christoffersen e Z2, o semáforo e o gráfico. Se o Z2 rejeitar com $Z_2 < 0$, aplica-se a mesma correção $k = 1 - Z_2$ ao ES da carteira.
* **Limitações:** com 5 anos de preços e a janela de 756 dias, sobram cerca de 500 dias de previsões (2 anos), sem nenhuma crise: os testes só apanham falhas grandes. Pesos constantes no passado não são a história real da carteira.

### D. Validação
* `resumo_metodos` e `validado` contra um exemplo feito à mão.
* A previsão da carteira é igual a correr `next_day_forecast` diretamente sobre $\sum_i w_i r_i$; o ES da carteira fica abaixo da soma ponderada dos ES isolados (diversificação); com uma posição de histórico curto, devolve "sem previsão".
* Com um ES sintético 1,4× abaixo do real, o ES corrigido por $k = 1 - Z_2$ dá $Z_2 = 0$ (erro < $10^{-12}$).
* A previsão diária rápida difere da do backtest completo em menos de 1% (só muda o ponto de partida do otimizador do GARCH).
* No backtest da carteira, as exceções contadas na série coincidem com as da tabela de testes.

---

## 🌊 14. Dependência de cauda: quem cai com quem (Fase A)

Código: `src/models/cauda.py` · parâmetros: `CAUDA_*` · testes: `tests/test_cauda.py` · saídas: `tail_dependence.csv` e `tail_meta.json`. Segue McNeil, Frey e Embrechts (2005), capítulo 5.

### A. Porque não basta a correlação
A correlação linear mede o andamento conjunto em dias normais. Para o contágio interessa outra pergunta: quando um setor tem um dia muito mau, o outro também tem? Na cópula de Gauss, a resposta no limite é "não", para qualquer correlação abaixo de 1. Um mapa feito só de correlações subestima por isso as quedas conjuntas.

### B. Duas medidas
* **Quedas conjuntas observadas:** com a cauda de cada série definida como os $q$ = 5% piores dias (retornos simples, 3 anos),
  $$\hat p_{AB} = \frac{\#\{t : A_t \text{ e } B_t \text{ na cauda}\}}{\#\{t : A_t \text{ na cauda}\}}$$
  É simétrica, porque as duas caudas têm a mesma probabilidade $q$. Sob independência vale cerca de 5%.
* **Coeficiente de dependência de cauda da cópula t:**
  $$\lambda = 2\, t_{\nu+1}\!\left(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right)$$
  É o limite da probabilidade condicional quando a cauda fica cada vez mais extrema. A cópula é ajustada assim:
  1. resíduos padronizados de um GJR-GARCH(1,1)-t por série, para tirar os clusters de volatilidade;
  2. $\rho_{ij} = \sin(\pi \tau_{ij}/2)$ a partir do tau de Kendall de cada par, com a matriz corrigida para ser definida positiva;
  3. um $\nu$ comum por máxima verosimilhança nas pseudo-observações (postos a dividir por $n + 1$).

### C. No mapa de risco
* As ligações do mapa passam a ser, por omissão, a árvore geradora mínima das **quedas conjuntas** (distância $1 - \hat p$), entre os 11 setores e entre as 10 empresas de cada setor. A correlação fica como alternativa.
* Cada ligação mostra a frequência de quedas conjuntas, o $\lambda$ e a correlação. A ficha do setor e a da empresa dizem com quem cada um cai junto.
* O separador Minha carteira usa a mesma espinha.

### D. Resultados (dados até 2026-11-11)
* **Setores:** $\nu$ = 9,6. Os pares que mais caem juntos são XLY–XLC (54%), XLF–XLY (51%), XLI–XLB e XLI–XLY (49%), contra 5% por acaso.
* **O que a correlação esconde:** o XLK–XLC está no top 10 das quedas conjuntas, mas é só o 17.º par por correlação (0,54). O centro do mapa muda: pelas quedas conjuntas é o XLY (consumo discricionário); pela correlação eram o XLI e o XLRE.
* O $\lambda$ é baixo (0,11 a 0,21 nos pares de topo): com $\nu$ perto de 10, a dependência no limite extremo é moderada. A frequência observada a 5% é bem mais alta, porque mede uma cauda menos extrema.

### E. Validação
* `lambda_t` reproduz os 11 valores da tabela 5.1 do livro (erro < 0,005).
* A verosimilhança da cópula t é igual à calculada com `scipy.stats.multivariate_t`.
* Com dados simulados de uma t bivariada ($\rho$ = 0,5, $\nu$ = 4), o ajuste recupera $\rho$ (±0,04), $\nu$ (±1,2) e $\lambda$; com dados gaussianos dá $\nu$ > 25 e $\lambda$ < 0,03.
* Quedas conjuntas: séries iguais dão 1, opostas dão 0 e independentes dão cerca de $q$.

### F. Limitações
* **Poucos dias de cauda:** 5% de 751 dias são cerca de 38 dias por série. Cada frequência tem um erro-padrão de uns 8 pontos percentuais; diferenças pequenas entre pares não são significativas.
* **Três anos sem crise sistémica:** a janela não inclui 2008 nem 2020. A dependência de cauda numa crise a sério é provavelmente maior.
* **Um só $\nu$ por grupo:** a cópula t tem a mesma cauda para todos os pares e é simétrica (quedas e subidas conjuntas iguais). Nos dados reais, as quedas conjuntas costumam ser mais fortes do que as subidas.
* **A árvore deixa ligações de fora:** mostra as N − 1 mais fortes que ligam tudo; a matriz completa está no cockpit.

---

## 📚 Referências
* Acerbi, C. e Székely, B. (2014). *Backtesting Expected Shortfall*. Risk, dezembro de 2014.
* Barone-Adesi, G., Giannopoulos, K. e Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. *Journal of Futures Markets*, 19(5), 583–602.
* Basel Committee on Banking Supervision (1996). *Supervisory framework for the use of "backtesting" in conjunction with the internal models approach to market risk capital requirements*.
* Basel Committee on Banking Supervision (2019). *Minimum capital requirements for market risk* (FRTB, d457).
* Christoffersen, P. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862.
* Glosten, L., Jagannathan, R. e Runkle, D. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779–1801.
* Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
* Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73–84.
* Ledoit, O. e Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
* McNeil, A., Frey, R. e Embrechts, P. (2005). *Quantitative Risk Management: Concepts, Techniques and Tools*. Princeton University Press.
* Mantegna, R. (1999). Hierarchical structure in financial markets. *European Physical Journal B*, 11, 193–197.
* Newey, W. e West, K. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
* Newey, W. e West, K. (1994). Automatic lag selection in covariance matrix estimation. *Review of Economic Studies*, 61(4), 631–653.
* Rockafellar, R. T. e Uryasev, S. (2002). Conditional value-at-risk for general loss distributions. *Journal of Banking & Finance*, 26(7), 1443–1471.
* Tasche, D. (2000). *Risk contributions and performance measurement*. Working paper, TU München.
