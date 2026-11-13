"""Painel do motor de cenários (Fase 3): escolher um choque, simular e mostrar o efeito em setores e empresas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import CENARIO_HORIZONTES, MACRO_FACTORS, SHOCK_DIFF, SHOCK_LOGDIFF, SHOCK_MARKET
from src.models import cenarios as cn
from src.report import formato as fm
from paineis.comum import nome

NOMES_F = {f: MACRO_FACTORS[f]["curto"] for f in MACRO_FACTORS} | {cn.MERCADO: "Mercado (S&P 500)"}


@st.cache_resource(show_spinner=False)
def _modelo(versao: str, _dados: pd.DataFrame, _hold: pd.DataFrame) -> dict:
    return cn.ajustar(_dados.set_index("data").astype(float), _hold)


def _em_unidades(f: str, v: float) -> float:
    """Valor do slider → unidade da regressão: p.p. para os fatores em diferença, Δ log para os outros."""
    return v if f in MACRO_FACTORS and MACRO_FACTORS[f]["transf"] == "diff" else float(np.log1p(v / 100))


def _txt(f: str, x: float) -> str:
    return fm.pp(x, 2) if f in MACRO_FACTORS and MACRO_FACTORS[f]["transf"] == "diff" else fm.pct(np.expm1(x), 1, sinal=True)


def render(g: dict, versao: str) -> tuple[pd.Series | None, list[str]]:
    """Desenha os controlos e o resultado. Devolve (mediana por setor e empresa, fatores chocados) ou (None, [])."""
    dados, val = g["cen_sem"], g["cen_val"]
    with st.expander("🎛️ Simular um choque: fatores → setores → empresas (Monte Carlo)",
                     expanded=bool(st.session_state.get("cen_fatores"))):
        c1, c2 = st.columns([3, 2])
        escolhidos = c1.multiselect("Fatores a chocar", list(NOMES_F), format_func=NOMES_F.get, key="cen_fatores",
                                    placeholder="escolhe um ou mais fatores")
        hz = c2.radio("Horizonte", list(CENARIO_HORIZONTES), index=1, horizontal=True, key="cen_h")
        choque = {}
        cols = st.columns(max(len(escolhidos), 1))
        for col, f in zip(cols, escolhidos):
            if f == cn.MERCADO:
                lo, hi, d0, passo = SHOCK_MARKET
                v = col.slider(f"{NOMES_F[f]} (%)", lo, hi, d0, passo, key=f"cen_{f}")
            elif MACRO_FACTORS[f]["transf"] == "diff":
                lo, hi, d0, passo = SHOCK_DIFF
                v = col.slider(f"{NOMES_F[f]} (p.p.)", lo, hi, d0, passo, key=f"cen_{f}")
            else:
                lo, hi, d0, passo = SHOCK_LOGDIFF
                v = col.slider(f"{NOMES_F[f]} (%)", lo, hi, d0, passo, key=f"cen_{f}")
            choque[f] = _em_unidades(f, v)
        if not choque:
            st.caption("Escolhe um fator e a variação no horizonte. Os outros fatores reagem como reagiram em média nos "
                       "últimos 10 anos, os setores pela sua sensibilidade aos fatores e as empresas pela sensibilidade "
                       "ao seu setor. A teia passa a mostrar o efeito.")
            return None, []
        mod = _modelo(versao, dados, g["hold"])
        sims = cn.simular(mod, choque, CENARIO_HORIZONTES[hz])
        res = cn.resumo(sims)
        outros = [f for f in cn.FAT if f not in choque]
        st.markdown("**O que o modelo espera para os outros fatores** (mediana): "
                    + " · ".join(f"{NOMES_F[f]} {_txt(f, sims[f].median())}" for f in outros))
        rs = res.loc[mod["setores"]].sort_values("mediana")
        st.dataframe(pd.DataFrame({
            "Setor": [nome(s) for s in rs.index],
            f"Retorno em {hz} (mediana)": rs["mediana"].map(lambda v: fm.pct(v, 1, sinal=True)).values,
            "Os 5% piores": rs["q05"].map(lambda v: fm.pct(v, 1, sinal=True)).values,
            "Os 5% melhores": rs["q95"].map(lambda v: fm.pct(v, 1, sinal=True)).values,
            "Probabilidade de perder": rs["p_perda"].map(lambda v: fm.pct(v, 0)).values,
        }), hide_index=True, width="stretch")
        txt_v = ""
        if val is not None:
            v = val.groupby("condicao").apply(lambda d: ((d["mediana"] - d["real"]).mean(), d["dentro_90"].mean()),
                                              include_groups=False)
            txt_v = (f" Testado em 2008, 2020 e 2022 com os dados de antes de cada crise: com os fatores macro reais, a "
                     f"perda prevista ficou em média {fm.pct(abs(v['macro'][0]), 0)} abaixo da real; com o mercado real "
                     f"também, o desvio desce para {fm.pct(abs(v['macro_mercado'][0]), 0)}. Só "
                     f"{fm.pct(v['macro_mercado'][1], 0)} dos setores ficaram dentro do intervalo de 90%: nas crises, os "
                     "intervalos são estreitos demais.")
        st.caption(f"{len(sims):,} cenários".replace(",", " ") + f", estimados em {mod['semanas']} semanas "
                   f"({mod['inicio'].date()} → {mod['fim'].date()}). O mercado é um fator: se não o escolheres, cai ou "
                   "sobe com o que costuma acompanhar o choque." + txt_v + " Na teia, a cor dos setores e das empresas "
                   "passa a ser a mediana do cenário (vermelho perde, verde ganha).")
        return res["mediana"].drop(cn.FAT), list(choque)
