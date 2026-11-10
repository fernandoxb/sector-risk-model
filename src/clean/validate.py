"""Bot Inspetor: valida os preços brutos e grava a versão limpa.

- Contrato pandera: preços positivos, datas únicas e ordenadas.
- Forward-fill de no máximo 1 dia; falhas maiores ficam como NaN e são reportadas.
- Retornos diários acima de ±25% são sinalizados (não apagados).
Saídas: data/clean/prices.parquet e data/clean/quality_report.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import CLEAN, MAX_ABS_RET, RAW, TICKERS  # noqa: E402

schema = pa.DataFrameSchema(
    {t: pa.Column(float, pa.Check.gt(0), nullable=True) for t in TICKERS},
    index=pa.Index("datetime64[ns]", unique=True, name="date"),
    checks=pa.Check(lambda df: df.index.is_monotonic_increasing, error="datas fora de ordem"),
    strict=True,
)


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
    (CLEAN / "quality_report.md").write_text("\n".join(lines) + "\n")
    print(f"[inspetor] contrato OK; relatório em {CLEAN / 'quality_report.md'}")
    return out


if __name__ == "__main__":
    main()
