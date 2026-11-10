---
tipo: documentacao-metodologica
projeto: Modelo de Risco Setorial Interativo
versao: 2.2
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

## 🧪 6. Backtesting do VaR (Sprint 2)

Código: `src/models/backtest.py` (funções puras) · parâmetros: `config/settings.py` (`BT_*`) · testes: `tests/test_backtest.py`.
Saídas: `data/features/var_backtest.csv` (resultados) e `data/features/var_backtest_series.parquet` (séries diárias para o cockpit).

### A. Previsão do VaR sem look-ahead
O VaR a 1 dia do dia $t$ é previsto **só com dados até $t-1$**, em janela móvel de $W$ = `CVAR_YEARS` × 252 = 756 dias úteis, para os níveis $\alpha \in \{99\%, 97{,}5\%\}$. Os dois métodos usam retornos simples e dão o VaR em perda positiva:

* **Histórico:** $\text{VaR}_t = -Q_{1-\alpha}(r_{t-W}, \dots, r_{t-1})$, o quantil empírico (interpolação linear, igual a `risk_metrics.hist_var`).
* **GJR-GARCH(1,1)-t:** estimado em retornos logarítmicos $x_t = \ln(1 + r_t)$ (em %), com média constante:
  $$x_t = \mu + \varepsilon_t,\quad \varepsilon_t = \sigma_t z_t,\quad z_t \sim t_\nu \text{ padronizada}$$
  $$\sigma_t^2 = \omega + (\alpha_1 + \gamma\, \mathbb{1}_{\varepsilon_{t-1}<0})\,\varepsilon_{t-1}^2 + \beta\, \sigma_{t-1}^2$$
  Os parâmetros são re-estimados no **1.º dia útil de cada mês** (`BT_REFIT_FREQ`) com as $W$ observações anteriores. Entre re-estimações, os parâmetros ficam fixos e $\sigma_t$ é atualizada todos os dias pela recursão. O quantil logarítmico passa a retorno simples pela transformação exata (monótona) $r = e^x - 1$:
  $$\text{VaR}_t = -\left(e^{\mu + \sigma_t\, q_{1-\alpha}(\nu)} - 1\right)$$

Os dois métodos são comparados na mesma amostra: os dias em que ambos têm previsão (a partir de 2008 nos setores com história desde 2005; XLRE e XLC começam mais tarde).

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

### G. Validação
* O Kupiec e o Christoffersen são comparados com o cálculo feito à mão com `math.log`, e o p-valor com $\operatorname{erfc}(\sqrt{LR/2})$.
* O VaR histórico é comparado com `risk_metrics.hist_var` na janela $[t-W, t-1]$.
* O VaR GARCH é comparado com a recursão GJR escrita à mão a partir dos parâmetros estimados.
* **Sem look-ahead:** alterar o retorno do dia $t$ e de todos os seguintes não muda nenhum VaR até ao dia $t$ (testado num dia de re-estimação e num dia a meio do mês).
* Reutilizar os parâmetros do mês anterior como ponto de partida do otimizador altera o VaR GARCH em no máximo 0,1% e não muda nenhuma exceção (verificado no XLK).

### H. Leitura dos resultados
O VaR histórico tende a falhar o teste de independência: a janela de 3 anos reage devagar e as exceções aparecem em grupos nas crises. O GARCH reage à volatilidade e normalmente passa a independência, mas pode subestimar a cauda (mais exceções do que as esperadas). A cobertura e a independência medem coisas diferentes, por isso o teste condicional resume as duas.
