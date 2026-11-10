"""Scorecard de risco por setor, 0–100 (Sprint 4b).

Funções puras (sem Streamlit, sem I/O) — o `main()` no fim é o único ponto com I/O:
  percentile_norm      valor → [0, 1] pelo percentil entre os setores (0 = menor, 1 = maior)
  stress_loss          perda média nos cenários de stress, com peso reduzido para linhas com proxy
  momentum             nº de exceções ao VaR oficial nos últimos `days` dias
  raw_components       junta as cinco fontes numa tabela de valores em bruto por setor
  normalize            componentes em [0, 1] (o regime já vem em [0, 1] e não é re-normalizado)
  weighted             média ponderada de um subconjunto de componentes
  scorecard            score 0–100, ajuste de risco de modelo, impacto e probabilidade (1–5)
  sensitivity          como muda o top 3 quando cada peso varia ±x% (re-normalizando)

Componentes (maior = mais risco):
  cauda       ES 97,5% oficial (FHS) de amanhã                       official_risk.csv
  sistemico   % do CVaR da carteira / peso                           risk_contribution.csv
  stress      perda média nos cenários históricos                    stress_tests.csv
  regime      percentil da vol 63d na própria história               sector_metrics.csv
  momento     exceções ao VaR 97,5% oficial nos últimos 60 dias      var_backtest_series.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (FEATURES, MODEL_RISK_LEVEL, MODEL_RISK_MULT,  # noqa: E402
                             MOMENTUM_DAYS, OFFICIAL_VAR_METHOD, SCORE_IMPACT, SCORE_PROBABILITY,
                             SCORE_SENSITIVITY, SCORE_WEIGHTS, SECTORS, STRESS_PROXY_WEIGHT)
from src.models.risk_metrics import quintile_score  # noqa: E402  (mesma regra 1–5 do mapa)

COMPONENTS = ("cauda", "sistemico", "stress", "regime", "momento")
ALREADY_UNIT = ("regime",)        # já é um percentil em [0, 1]


# ----------------------------------------------------------------------------- componentes
def percentile_norm(s: pd.Series) -> pd.Series:
    """(posição - 1) / (n - 1): o menor vale 0, o maior vale 1; empates ficam com a posição média."""
    if len(s) == 1:
        return pd.Series(0.5, index=s.index)
    return (s.rank(method="average") - 1) / (len(s) - 1)


def stress_loss(stress: pd.DataFrame, tickers: list[str], proxy_weight: float = STRESS_PROXY_WEIGHT) -> pd.Series:
    """Média ponderada da perda por cenário: peso 1, ou `proxy_weight` quando a linha é proxy (β × SPY)."""
    s = stress[stress["serie"].isin(tickers)].copy()
    s["w"] = np.where(s["proxy"].astype(bool), proxy_weight, 1.0)
    g = s.assign(wl=s["w"] * s["perda"]).groupby("serie")
    return (g["wl"].sum() / g["w"].sum()).reindex(tickers)


def momentum(series: pd.DataFrame, tickers: list[str], method: str = OFFICIAL_VAR_METHOD,
             level: float = 0.975, days: int = MOMENTUM_DAYS) -> pd.Series:
    """Exceções (perda > VaR previsto) nos últimos `days` dias de backtest de cada setor."""
    col = f"var_{method}_{level}"
    out = {}
    for t in tickers:
        s = series[series["setor"] == t].sort_values("data").tail(days)
        out[t] = int((-s["ret"] > s[col]).sum())
    return pd.Series(out)


def raw_components(official: pd.DataFrame, rc: pd.DataFrame, stress: pd.DataFrame, metrics: pd.DataFrame,
                   series: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Valores em bruto (maior = mais risco). Índices por ticker em `official`, `rc` e `metrics`."""
    return pd.DataFrame({
        "cauda": official["es975"].reindex(tickers),
        "sistemico": (rc["pct_cvar"] / rc["peso"]).reindex(tickers),
        "stress": stress_loss(stress, tickers),
        "regime": metrics["vol_pct"].reindex(tickers).astype(float),
        "momento": momentum(series, tickers),
    })


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: raw[c] if c in ALREADY_UNIT else percentile_norm(raw[c]) for c in raw.columns})


def weighted(comp: pd.DataFrame, weights: dict[str, float], keys: tuple[str, ...] | None = None) -> pd.Series:
    """Σ w_k c_k / Σ w_k sobre `keys` (todas as componentes por omissão)."""
    keys = tuple(keys or weights)
    w = pd.Series({k: weights[k] for k in keys})
    return comp[list(keys)].mul(w, axis=1).sum(axis=1) / w.sum()


# ----------------------------------------------------------------------------- score
def scorecard(raw: pd.DataFrame, traffic: pd.Series, weights: dict[str, float] = SCORE_WEIGHTS,
              mult: dict[str, float] = MODEL_RISK_MULT) -> pd.DataFrame:
    """Score 0–100 = 100 · Σ w_k c_k; score_final = score × multiplicador do semáforo (máx. 100)."""
    if not np.isclose(sum(weights.values()), 1):
        raise ValueError("SCORE_WEIGHTS tem de somar 1")
    comp = normalize(raw)
    score = 100 * weighted(comp, weights)
    sem = traffic.reindex(raw.index)
    m = sem.map(mult)
    out = pd.concat([raw.add_prefix("bruto_"), comp], axis=1)
    out["score"] = score
    out["semaforo"] = sem
    out["mult_modelo"] = m
    out["score_final"] = (score * m).clip(upper=100)
    out["risco_modelo"] = sem != "verde"
    out["impacto"] = quintile_score(weighted(comp, weights, SCORE_IMPACT))
    out["probabilidade"] = quintile_score(weighted(comp, weights, SCORE_PROBABILITY))
    out["rank"] = out["score_final"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("score_final", ascending=False)


def _top(raw: pd.DataFrame, traffic: pd.Series, weights: dict[str, float], n: int = 3) -> list[str]:
    return list(scorecard(raw, traffic, weights).index[:n])


def sensitivity(raw: pd.DataFrame, traffic: pd.Series, weights: dict[str, float] = SCORE_WEIGHTS,
                factor: float = SCORE_SENSITIVITY, n: int = 3) -> pd.DataFrame:
    """Para cada peso × (1 ± factor), re-normaliza para somar 1 e devolve o novo top n."""
    base = _top(raw, traffic, weights, n)
    rows = []
    for k in weights:
        for sign in (-1, +1):
            w = dict(weights)
            w[k] = weights[k] * (1 + sign * factor)
            tot = sum(w.values())
            w = {kk: v / tot for kk, v in w.items()}
            top = _top(raw, traffic, w, n)
            rows.append({"componente": k, "variacao": f"{sign * factor:+.0%}", "peso_novo": w[k],
                         "top": " · ".join(top), "mesmo_top": set(top) == set(base),
                         "mesma_ordem": top == base})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- I/O
def main() -> Path:
    tickers = list(SECTORS)
    official = pd.read_csv(FEATURES / "official_risk.csv", index_col="setor")
    rc = pd.read_csv(FEATURES / "risk_contribution.csv", index_col=0)
    stress = pd.read_csv(FEATURES / "stress_tests.csv")
    metrics = pd.read_csv(FEATURES / "sector_metrics.csv", index_col=0)
    series = pd.read_parquet(FEATURES / "var_backtest_series.parquet")
    bt = pd.read_csv(FEATURES / "var_backtest.csv")
    off = bt[(bt["metodo"] == OFFICIAL_VAR_METHOD) & np.isclose(bt["nivel"], MODEL_RISK_LEVEL)]
    traffic = off.set_index("setor")["semaforo"]

    raw = raw_components(official, rc, stress, metrics, series, tickers)
    sc = scorecard(raw, traffic)
    sens = sensitivity(raw, traffic)

    out = FEATURES / "scorecard.csv"
    sc.rename_axis("setor").to_csv(out, float_format="%.6f")
    sens.to_csv(FEATURES / "scorecard_sensitivity.csv", index=False, float_format="%.4f")

    # acrescenta ao JSON do mapa (o Repórter embebe-o no HTML)
    mj = FEATURES / "risk_map_data.json"
    payload = json.loads(mj.read_text())
    payload["scorecard"] = {
        "metodo": {"pesos": SCORE_WEIGHTS, "impacto": list(SCORE_IMPACT), "probabilidade": list(SCORE_PROBABILITY),
                   "modelo_oficial": OFFICIAL_VAR_METHOD, "mult_risco_modelo": MODEL_RISK_MULT,
                   "data_previsao": str(official["data_previsao"].iloc[0])},
        "setores": {t: {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else
                            bool(v) if isinstance(v, (bool, np.bool_)) else
                            int(v) if isinstance(v, (int, np.integer)) else v)
                        for k, v in row.items()} for t, row in sc.iterrows()},
    }
    mj.write_text(json.dumps(payload, ensure_ascii=False, indent=1))

    print(f"[scorecard] {len(sc)} setores → {out.name} (+ scorecard no risk_map_data.json)")
    cols = ["rank", *COMPONENTS, "score", "semaforo", "score_final", "impacto", "probabilidade"]
    print(sc[cols].round(3).to_string())
    print(sens.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
