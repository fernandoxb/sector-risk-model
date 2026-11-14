---
lingua: en
capa: Sector Risk Model · short summary · code, data and full methodology (in Portuguese): github.com/fernandoxb/sector-risk-model
rodape: Sector Risk Model · summary
---

# Sector Risk Model: a short summary

**What it is.** An end-to-end market-risk system for the 11 SPDR sector ETFs and their 110 largest holdings, built as a portfolio project for a Master's in Risk Management. It forecasts next-day and multi-day VaR and Expected Shortfall, tests those forecasts out of sample, measures how sectors and companies fall together, and simulates how a macro shock spreads from factors to sectors to single stocks. A scheduled pipeline refreshes everything every trading day; a public cockpit shows it: **[sector-risk-model.streamlit.app](https://sector-risk-model.streamlit.app)**.

**Data.** Daily adjusted prices from Yahoo Finance (sector ETFs and SPY since 2005, the 110 companies since 2005), macro factors from FRED and Yahoo (10-year Treasury, 10-year breakeven inflation, high-yield and Baa credit spreads, VIX, WTI oil, dollar index). Simple returns for tail risk, log returns for GARCH. Every run is validated by a data contract (pandera) before any model runs.

## 1. Which VaR/ES model survives the backtest?

Four methods forecast 1-day VaR and ES using only data up to the day before, on a rolling 3-year window: historical simulation, GJR-GARCH(1,1) with Student-t innovations, Filtered Historical Simulation (FHS) and conditional EVT (GARCH plus a generalised Pareto tail; McNeil and Frey, 2000). About 46,000 forecasts per confidence level, 2008–2026, are tested with Kupiec, Christoffersen, the Acerbi–Székely Z2 test for ES (simulated p-values), the McNeil–Frey ES test and Basel traffic lights.

| Series passing Kupiec, Christoffersen and Z2 at 97.5% | Model choice, 2008–2018 | Confirmation, 2019–2026 | 109 companies (since 2008) |
|---|---|---|---|
| **GARCH-EVT** (official model) | **9 of 9** | **10 of 11** | **76** |
| GARCH-FHS | 6 of 9 | 9 of 11 | 55 |
| GJR-GARCH-t | 0 of 9 | 0 of 11 | 26 |
| Historical | 0 of 9 | 3 of 11 | 8 |

* The method is chosen on 2008–2018 only and confirmed on 2019–2026, which the choice never saw, and on single stocks.
* The historical method reacts late to crises (clustered exceptions); GARCH-t has too thin a tail; combining GARCH volatility with the empirical or Pareto tail of the residuals fixes both.
* With a Holm correction for the 88 tests per method, no rejection of the EVT model survives, in sectors or companies.
* Where the ES is still underestimated (17 companies), the forecast is scaled by the factor that would have set Z2 to zero in the backtest (on average ×1.22).
* **Longer horizons:** 10- and 21-day VaR/ES by simulating GARCH paths. In non-overlapping 10-day blocks since 2008, 121 exceptions at 97.5% against 115 expected; the square-root-of-time rule does about as well on frequency (122), but differs day to day when volatility is far from its average.

## 2. Who falls with whom?

Correlation describes normal days; contagion is about bad days. For every pair of sectors the system counts how often both are in their 5% worst days, and fits a t copula (Kendall's tau, ν by maximum likelihood) to GARCH residuals (QRM, chapter 5).

* Consumer discretionary and communication services fall together on **54%** of their worst days (5% if independent). Over 10 years, with COVID-19 in the window, the average pair rises from 28% to 44%: crises are when everything falls together.
* Sectors fall together more often than they rise together (27.9% against 24.2%). A symmetric t copula on GARCH residuals simulated only 22.9%; a skewed t copula on returns, with its asymmetry set by the method of moments (γ = −0.15), simulates 26.8%.
* The cockpit's risk web links sectors by these joint falls and can switch to 1, 5 or 10-year windows.

## 3. What moves each sector, and each company?

* Weekly regressions of each sector on six macro factors with Newey–West errors: 34 of 66 links are significant one factor at a time; with all six factors in one regression, 28 survive and 12 vanish (rates → energy was really oil and inflation).
* Each company's daily variance splits into market, sector and own risk: median 13%, 21% and **56%**. Energy and utilities are mostly sector (Exxon: 83%); McDonald's and Verizon are over 90% own risk.

## 4. What if a factor moves? A Monte Carlo scenario engine

The user picks a shock (e.g. VIX +50% in a month). Unshocked factors follow the conditional normal distribution given the shock; sectors respond through a joint regression on the six macro factors plus the market, with bootstrapped weekly residuals widened when the shock is unusually intense and with parameter uncertainty (50 bootstrap re-estimations); companies respond through their beta to their own sector ETF. 5,000 scenarios run live in about 0.05 s, and the web is coloured by the result.

| Out-of-sample test (model fitted before each crisis) | Macro factors imposed | Macro factors and market imposed |
|---|---|---|
| Lehman 2008: predicted minus realised sector loss, average | +17.7 pts | +0.5 pts |
| COVID-19 2020 | +16.8 pts | +3.5 pts |
| Rate hikes 2022 | +10.4 pts | −7.7 pts |
| Sector-crises inside the 90% model interval (≈28 of 31 expected) | 7 of 31 | 16 of 31 |

**Reading:** given the market move, the engine places sectors well; from macro factors alone it understates crises by about 15 points, because the factor–market link estimated in calm years misses panic. The 90% intervals are **model intervals, not confidence intervals**: in crises they are too narrow, and the cockpit shows the measured coverage next to every result.

## 5. Portfolio tools

* **Any portfolio** of the 110 companies, the ETFs, SPY or any Yahoo ticker (sector assigned automatically): Euler contributions to volatility and CVaR, 1-day, 10-day and 1-month forecasts with their own backtest, historical stress tests (2008, 2020, 2022) and the scenario engine's result for the portfolio.
* **Hierarchical Risk Parity** on the joint-fall distance, with CVaR as the cluster risk. Re-estimated every 63 days on the previous 504 only (out of sample), on an example portfolio: CVaR 2.56% against 3.47% for the user's weights and 2.95% for equal weights.
* **0–100 sector scorecard:** tail, systemic, stress, regime and momentum; the tail compares each sector both with the others and with its own history; Dirichlet draws of the weights give each sector a rank interval (XLK is first under every draw); since the intervals overlap, the cockpit shows groups instead of ranks: XLK; then XLY, XLE, XLI and XLF; then the other six.

## 6. Engineering

Prefect flow with one task per step (collector, validator, models, backtests, scenarios, alerts, weekly report, publisher), scheduled with launchd every trading day at 22:30 Lisbon time on the NYSE calendar. Alerts and a weekly report go to an Obsidian vault; a weekly snapshot is published to a separate Git branch so the main branch holds only code. 206 pytest tests, each new calculation checked against an independent implementation, GitHub Actions CI.

## Limitations

US sector ETFs and their largest holdings only. Several results rest on few crisis days (about 38 tail days per pair over 3 years). The scenario engine is linear and understates crises without the market move. HRP and the company ES correction are evaluated on relatively short or in-sample periods. It is an academic project, not investment advice.

*Full methodology, formulas and every validation, in Portuguese:* `docs/nota-metodologica.pdf` *(19 sections).*
