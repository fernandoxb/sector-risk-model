---
tipo: documentacao-metodologica
projeto: Modelo de Risco Setorial Interativo
versao: 3.6
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
* **Camada micro:** as 4 ações mais relevantes do setor selecionado (`config/settings.py → SECTOR_PROFILE`). O CVaR de cada ação é comparado com o CVaR do ETF do setor **no mesmo período**. Se for maior, a ação recebe o diagnóstico ⚠️, ou seja, está a acrescentar risco ao setor.
* **Correlação cruzada micro:** um mapa de calor mostra a correlação entre as ações do setor selecionado.

---

## 🏦 4. Regime de juros (Treasury 10Y via FRED)

* Série `DGS10`, lida pela `fredapi` quando `FRED_API_KEY` está definida e, caso contrário, pelo CSV público do FRED.
* **Regime:** variação do 10Y nos últimos 63 dias úteis. É *subida* se ≥ +0,25 p.p., *descida* se ≤ −0,25 p.p. e *estável* nos restantes casos.
* **Penalização:** para os setores de crescimento (XLK, XLC, XLY), aplica-se
  $\text{CVaR}_{aj} = \text{CVaR} \times (1 + \lambda \cdot \max(\Delta y, 0))$,
  em que $\lambda$ é a penalização por +1 p.p. (ajustável) e $\Delta y$ é a variação real somada a um choque opcional definido no painel. A penalização é contínua: aplica-se a qualquer $\Delta y > 0$, mesmo quando o rótulo do regime é *estável*.
* **Beta de juros (informativo):** sensibilidade do retorno diário de cada setor a +1 p.p. no 10Y. Serve para confirmar empiricamente se o setor se comporta como um ativo de longa duração.

---

## 🗂️ 5. Persistência em Markdown (Obsidian)

O botão **Exportar estado de hoje** grava `<cofre>/Risco Setorial/AAAA-MM-DD.md` com:
- frontmatter com os parâmetros e o regime de juros, para consultas Dataview;
- a tabela de setores, a lista de ligações da MST e o drill-down do setor selecionado, com todos os tickers como `[[Ticker]]` para gerar backlinks.

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

### K. Modelo oficial e previsão para o próximo dia útil (Sprint 4a)
O backtest escolhe o modelo oficial: `OFFICIAL_VAR_METHOD = "fhs"`. A 97,5%, o FHS passa o Kupiec em 11/11 setores, o Christoffersen em 10/11 e o Z2 em 10/11. O histórico só passa a independência em 1/11 e o GARCH-t passa o Kupiec em 1/11 e o Z2 em 0/11. A tabela completa está no comentário de `config/settings.py`.

Todos os dias, o pipeline grava em `data/features/official_risk.csv` o **VaR 99%, o VaR 97,5% e o ES 97,5% para o próximo dia útil** (`OFFICIAL_MEASURES`), com todos os dados disponíveis até hoje.

**Sem lógica duplicada:** a série de cada setor ganha um dia extra no fim, o dia útil seguinte (seg–sex, sem calendário de feriados), com retorno 0. Corre-se o mesmo `sector_forecasts` do backtest: a última linha é a previsão de amanhã e as restantes são o backtest. O valor 0 é só um marcador, porque a previsão de um dia usa apenas dados até à véspera. Se esse dia for o primeiro de um mês, o GARCH é re-estimado com os dados até hoje, como no backtest.

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
* **Uma task por bot:** Coletor → Inspetor → Modelador → Backtester → Carteira → Scorecard → Ligações → Repórter → Snapshot → Sentinela → Relatório.
* **Snapshot (Sprint 6):** copia para `app/cockpit/snapshot/` (versionado, cerca de 2,7 MB) os preços de 3 anos dos ETFs e das ações do microscópio, o Treasury 10Y e os resultados que o cockpit lê. Assim o cockpit corre sem rede, por exemplo no Streamlit Community Cloud. As séries do backtest são gravadas em float32, o que dá um erro relativo de cerca de $10^{-7}$, sem efeito nas exceções mostradas.
* **Coletor:** 2 novas tentativas, 5 minutos entre elas, e cache por data. Dois runs no mesmo dia descarregam os dados uma só vez.
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

* **Exceções:** precisam só do próprio dia, por isso também saem no primeiro run.
* **Alertas.md:** o Sentinela escreve `Risco Setorial/Alertas.md`, com uma secção por data e a mais recente no topo. Cada alerta tem campos inline do Dataview (`[severidade:: alta]`). Repetir o run do mesmo dia reescreve a secção, sem duplicar.
* **Notificação:** só os alertas de severidade alta geram uma notificação macOS (`osascript`).
* **Limitação:** o nível P × I usa quintis entre os 11 setores, por isso um setor pode mudar de nível sem que o seu score se mexa, apenas porque outro mudou de posição. Ex.: XLI a 2026-11-10, P×I 16 → 20 com score 69,3 → 69,2.

### C. Relatório semanal (`src/report/weekly.py`)
* **Quando:** sai quando a data dos dados é uma sexta (ou com `--semanal`) e é gravado em `Risco Setorial/Semana AAAA-Www.md`.
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

## 📚 Referências
* Acerbi, C. e Székely, B. (2014). *Backtesting Expected Shortfall*. Risk, dezembro de 2014.
* Barone-Adesi, G., Giannopoulos, K. e Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. *Journal of Futures Markets*, 19(5), 583–602.
* Basel Committee on Banking Supervision (1996). *Supervisory framework for the use of "backtesting" in conjunction with the internal models approach to market risk capital requirements*.
* Basel Committee on Banking Supervision (2019). *Minimum capital requirements for market risk* (FRTB, d457).
* Christoffersen, P. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862.
* Glosten, L., Jagannathan, R. e Runkle, D. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779–1801.
* Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73–84.
* Ledoit, O. e Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
* Mantegna, R. (1999). Hierarchical structure in financial markets. *European Physical Journal B*, 11, 193–197.
* Newey, W. e West, K. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
* Newey, W. e West, K. (1994). Automatic lag selection in covariance matrix estimation. *Review of Economic Studies*, 61(4), 631–653.
* Rockafellar, R. T. e Uryasev, S. (2002). Conditional value-at-risk for general loss distributions. *Journal of Banking & Finance*, 26(7), 1443–1471.
* Tasche, D. (2000). *Risk contributions and performance measurement*. Working paper, TU München.

