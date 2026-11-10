---
tipo: documentacao-metodologica
projeto: Modelo de Risco Setorial Interativo
versao: 2.1
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
