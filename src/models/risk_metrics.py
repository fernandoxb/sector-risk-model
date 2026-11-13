"""Bot Modelador: métricas de risco por setor e visão conjunta.

Por setor (retornos logarítmicos diários):
  vol_1y      volatilidade realizada do último ano, anualizada
  vol_garch   volatilidade prevista para amanhã por GJR-GARCH(1,1) com t-Student, anualizada
  vol_pct     percentil da vol de 63 dias atual face à sua própria história (regime)
  var95/var99 VaR histórico a 1 dia (retornos simples, janela CVAR_YEARS), perda positiva
  cvar        Expected Shortfall histórico a 1 dia ao nível CVAR_LEVEL — mesma função do cockpit
  var99_garch VaR a 1 dia, 99%, condicional ao GARCH
  max_dd      drawdown máximo desde o início; dd_now = drawdown atual
  beta        beta face ao SPY (retornos simples, janela CVAR_YEARS)

Pontuações para o mapa (1-5):
  impacto       = quintil do CVaR entre os 11 setores (5 = cauda mais pesada)
  probabilidade = quintil do percentil de volatilidade atual (5 = regime mais tenso que 80% da história)

Saída: data/features/sector_metrics.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (CLEAN, CVAR_LEVEL, CVAR_YEARS, FEATURES, MARKET,  # noqa: E402
                             SECTORS, TRADING_DAYS, WIN_SHORT, WIN_YEAR)
from src.models.cockpit import cvar as es  # noqa: E402  (fonte única do CVaR)

warnings.filterwarnings("ignore")
ANN = np.sqrt(TRADING_DAYS)


def hist_var(r: pd.Series, q: float) -> float:
    """VaR histórico como perda positiva."""
    return float(-r.quantile(1 - q))


def drawdown(prices: pd.Series) -> pd.Series:
    return prices / prices.cummax() - 1


def garch(r: pd.Series) -> tuple[float, float, dict]:
    """GJR-GARCH(1,1)-t nos últimos 10 anos. Devolve (vol anualizada prevista, VaR99 1 dia, parâmetros)."""
    x = r.dropna().iloc[-10 * TRADING_DAYS:] * 100
    res = arch_model(x, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
    f = res.forecast(horizon=1, reindex=False)
    mu, var = f.mean.iloc[-1, 0], f.variance.iloc[-1, 0]
    q01 = res.model.distribution.ppf(0.01, res.params[["nu"]].values)
    var99 = (mu + np.sqrt(var) * q01) / 100
    params = {k: round(float(v), 4) for k, v in res.params.items()}
    return float(np.sqrt(var) / 100 * ANN), float(var99), params


def quintile_score(s: pd.Series) -> pd.Series:
    """Ranking em percentis → pontuação 1..5."""
    return np.ceil(s.rank(pct=True) * 5).clip(1, 5).astype(int)


def main() -> Path:
    prices = pd.read_parquet(CLEAN / "prices.parquet")
    rets = np.log(prices).diff()               # log: volatilidade e GARCH
    simple = prices.pct_change()               # simples: VaR, CVaR e beta (como o cockpit)
    asof_ts = prices.index.max()
    start_tail = asof_ts - pd.DateOffset(years=CVAR_YEARS)
    mkt_s = simple[MARKET]
    rows = {}
    for t, name in SECTORS.items():
        r = rets[t].dropna()
        s3 = simple[t].loc[simple.index > start_tail].dropna()
        roll = r.rolling(WIN_SHORT).std().dropna()
        vg, var99g, params = garch(r)
        both = pd.concat([s3, mkt_s.reindex(s3.index)], axis=1).dropna()
        beta = float(np.cov(both.iloc[:, 0], both.iloc[:, 1])[0, 1] / both.iloc[:, 1].var())
        dd = drawdown(prices[t].dropna())
        rows[t] = {
            "nome": name,
            "inicio": str(r.index.min().date()),
            "vol_1y": float(r.iloc[-WIN_YEAR:].std() * ANN),
            "vol_lr": float(r.std() * ANN),
            "vol_63d": float(roll.iloc[-1] * ANN),
            "vol_pct": float((roll <= roll.iloc[-1]).mean()),
            "vol_garch": vg,
            "var95": hist_var(s3, 0.95),
            "var99": hist_var(s3, 0.99),
            "cvar": es(s3, CVAR_LEVEL),
            "var99_garch": -var99g,
            "max_dd": float(dd.min()),
            "dd_now": float(dd.iloc[-1]),
            "beta": beta,
            "garch": params,
        }
    m = pd.DataFrame(rows).T
    m["impacto"] = quintile_score(m["cvar"].astype(float))
    m["probabilidade"] = np.ceil(m["vol_pct"].astype(float) * 5).clip(1, 5).astype(int)

    FEATURES.mkdir(parents=True, exist_ok=True)
    out = FEATURES / "sector_metrics.csv"
    m.drop(columns="garch").to_csv(out, float_format="%.6f")
    asof = str(prices.index.max().date())
    print(f"[modelador] métricas de {len(rows)} setores até {asof}")
    print(m[["vol_1y", "vol_garch", "vol_pct", "var99", "cvar", "max_dd", "beta", "impacto", "probabilidade"]]
          .astype(float).round(3).to_string())
    return out


if __name__ == "__main__":
    main()
