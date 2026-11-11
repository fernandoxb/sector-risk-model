"""Bot Inspetor: valida os preços e os fatores macro brutos e grava a versão limpa.

Preços
- Contrato pandera: preços positivos, datas únicas e ordenadas.
- Forward-fill de no máximo 1 dia; falhas maiores ficam como NaN e são reportadas.
- Retornos diários acima de ±25% são sinalizados (não apagados).
Fatores macro
- Contrato pandera: uma coluna por fator de MACRO_FACTORS, valores finitos, datas únicas e ordenadas.
- Níveis <= 0 em fatores com variação logarítmica são sinalizados (ex.: o WTI negativo de abril de 2020)
  e a data do último valor de cada fator é reportada (as séries do FRED chegam com 1 dia de atraso).
Se um contrato falhar, a exceção pára o pipeline: nada a jusante corre.
Saídas: data/clean/prices.parquet, data/clean/factors.parquet e data/clean/quality_report.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import CLEAN, MACRO_FACTORS, MAX_ABS_RET, RAW, TICKERS  # noqa: E402

schema = pa.DataFrameSchema(
    {t: pa.Column(float, pa.Check.gt(0), nullable=True) for t in TICKERS},
    index=pa.Index("datetime64[ns]", unique=True, name="date"),
    checks=pa.Check(lambda df: df.index.is_monotonic_increasing, error="datas fora de ordem"),
    strict=True,
)

factor_schema = pa.DataFrameSchema(
    {f: pa.Column(float, pa.Check(lambda s: np.isfinite(s.dropna()).all(), error="valor infinito"),
                  nullable=True) for f in MACRO_FACTORS},
    checks=[pa.Check(lambda df: isinstance(df.index, pd.DatetimeIndex), error="índice não é de datas"),
            pa.Check(lambda df: df.index.is_unique, error="datas repetidas"),
            pa.Check(lambda df: df.index.is_monotonic_increasing, error="datas fora de ordem")],
    strict=True,
)


def factor_report(f: pd.DataFrame, factors: dict = MACRO_FACTORS) -> list[str]:
    """Linhas do relatório de qualidade dos fatores (pura: só lê o DataFrame já validado)."""
    lines = ["## Fatores macro", "",
             "| Fator | Série | Início | Último valor | Observações | Níveis <= 0 (log) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for k, meta in factors.items():
        s = f[k].dropna()
        bad = s[s <= 0] if meta["transf"] == "logdiff" else s.iloc[:0]
        flag = ", ".join(f"{d.date()} ({v:.2f})" for d, v in bad.items()) or "—"
        lines.append(f"| {k} | {meta['serie']} | {s.index.min().date()} | {s.index.max().date()} | {len(s)} | {flag} |")
    return lines


def validate_factors() -> list[str]:
    raw = pd.read_parquet(RAW / "factors.parquet")
    factor_schema.validate(raw, lazy=True)
    raw.to_parquet(CLEAN / "factors.parquet")
    return factor_report(raw)


def main() -> Path:
    raw = pd.read_parquet(RAW / "prices.parquet")
    schema.validate(raw, lazy=True)

    # Só preencher buracos de 1 dia dentro da história de cada ticker
    clean = raw.copy()
    for t in TICKERS:
        first = clean[t].first_valid_index()
        clean.loc[first:, t] = clean.loc[first:, t].ffill(limit=1)

    rets = np.log(clean).diff()
    lines = ["# Relatório de qualidade dos dados", "",
             f"Período: {clean.index.min().date()} → {clean.index.max().date()} ({len(clean)} dias)", "",
             "| Ticker | Início | Dias | Falhas preenchidas | Falhas por resolver | Retornos > ±25% |",
             "| --- | --- | --- | --- | --- | --- |"]
    for t in TICKERS:
        first = raw[t].first_valid_index()
        span = raw.loc[first:, t]
        filled = int(span.isna().sum() - clean.loc[first:, t].isna().sum())
        left = int(clean.loc[first:, t].isna().sum())
        big = rets[t][rets[t].abs() > MAX_ABS_RET]
        flag = ", ".join(f"{d.date()} ({v:+.1%})" for d, v in big.items()) or "—"
        lines.append(f"| {t} | {first.date()} | {span.notna().sum()} | {filled} | {left} | {flag} |")

    CLEAN.mkdir(parents=True, exist_ok=True)
    out = CLEAN / "prices.parquet"
    clean.to_parquet(out)
    lines = lines[:2] + ["## Preços", ""] + lines[2:] + [""] + validate_factors()
    (CLEAN / "quality_report.md").write_text("\n".join(lines) + "\n")
    print(f"[inspetor] contratos de preços e fatores OK; relatório em {CLEAN / 'quality_report.md'}")
    return out


if __name__ == "__main__":
    main()
