# Contributing

Thanks for looking at the code. Issues and pull requests are welcome, in English or Portuguese.

## Run it

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                  # no network needed
.venv/bin/streamlit run app/cockpit/app.py  # works offline with the bundled snapshot
```

## Conventions

* **One source of truth:** every parameter (confidence levels, windows, thresholds, tickers, paths) lives in `config/settings.py`. No magic numbers elsewhere.
* **One function per metric:** search before adding a new one (the CVaR, for example, is `src/models/cockpit.py::cvar`).
* **Pure calculation functions** in `src/models/` (no Streamlit, no I/O), so they can be tested.
* **Risk conventions:** simple returns and positive losses for tail risk; log returns for volatility and GARCH.
* **Validate new calculations** against an independent implementation (numpy by hand against a library, or a known result) and add a test that fails if they break.
* **Language:** code comments, user interface and the methodological note are in European Portuguese; the README and these community files are in English.
* **Methodology changes** go into `docs/nota-metodologica.md`, with what changed and why.
* **Never commit data or secrets** (`data/`, `.env`, `FRED_API_KEY`). The public snapshot holds only base-100 indices and derived results.
