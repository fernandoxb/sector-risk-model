---
tipo: documentacao-metodologica
projeto: Modelo de Risco Setorial Interativo
versao: 2.3
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
