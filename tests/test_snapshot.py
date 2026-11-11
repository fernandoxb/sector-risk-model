"""Testes do snapshot do cockpit (src/report/snapshot.py): janelas e lista de ações."""
import pandas as pd

from src.report import snapshot as sn


def test_snapshot_janela_e_acoes():
    idx = pd.bdate_range("2020-01-01", "2026-11-10")
    p = pd.DataFrame({"A": range(len(idx))}, index=idx, dtype=float)
    w = sn.window(p, 3)
    assert w.index.min() > pd.Timestamp("2023-11-10") and w.index.max() == idx.max()
    r = sn.rates_window(pd.Series(1.0, index=idx), w.index.min(), 200)
    assert r.index.min() >= w.index.min() - pd.Timedelta(days=200) > r.index.min() - pd.Timedelta(days=5)
    acoes = sn.stock_tickers({"X": {"acoes": ["A", "B"]}, "Y": {"acoes": ["B", "C"]}})
    assert acoes == ["A", "B", "C"]
