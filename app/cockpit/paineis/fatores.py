"""Fatores macro: ligações medidas fator → setor (HAC).

Gerado a partir do antigo app.py: o código do separador é o mesmo; os dados do run chegam em `g`.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config.settings import CVAR_YEARS, LINK_T_MIN, MACRO_FACTORS
from src.report import formato as fm
from paineis.comum import RISCO, layout, nome


def render(g: dict) -> None:
    links = g["links"]
    setores = g["setores"]

    if links is None:
        st.info("Ainda não há ligações medidas. Corre `python run_pipeline.py --offline`.")
    else:
        st.subheader("Como os fatores macro mexem com cada setor")
        st.caption(f"Regressão semanal de cada setor em cada fator, nos últimos {CVAR_YEARS} anos, com erros-padrão "
                   f"HAC (Newey–West). Uma ligação é aceite quando |t| ≥ {fm.num(LINK_T_MIN, 0)}. "
                   "Lê-se como: **o que acontece ao setor quando o fator se move na direção adversa** "
                   "(juros, inflação, spreads, VIX, petróleo ou dólar a subir).")
        fat = list(MACRO_FACTORS)
        nomes_f = [MACRO_FACTORS[f]["curto"] for f in fat]
        L = links.set_index(["fator", "setor"])
        # efeito do movimento adverso: + = agrava o setor (t com o sinal trocado se subir é adverso)
        efeito = pd.DataFrame({f: [-MACRO_FACTORS[f]["adverso"] * L.loc[(f, t), "t"] for t in setores] for f in fat},
                              index=setores)
        aceite = pd.DataFrame({f: [bool(L.loc[(f, t), "aceite"]) for t in setores] for f in fat}, index=setores)
        txt = [[(fm.num(v, 1) if ok else f"({fm.num(v, 1)})") for v, ok in zip(rv, ra)]
               for rv, ra in zip(efeito.values, aceite.values)]
        hm_f = go.Figure(go.Heatmap(
            z=np.clip(efeito.values, -8, 8), x=nomes_f, y=[nome(t) for t in setores], zmid=0,
            colorscale=[[0, RISCO["verde"]], [0.5, "#131c2e"], [1, RISCO["vermelho"]]], text=txt, xgap=2, ygap=2,
            texttemplate="%{text}", colorbar=dict(title="t", thickness=12),
            hovertemplate="%{y} · %{x}<br>t do efeito adverso: %{text}<extra></extra>"))
        layout(hm_f, 480, "Efeito do movimento adverso (estatística t): vermelho agrava, verde atenua",
               yaxis=dict(autorange="reversed"), xaxis=dict(side="top", tickangle=0))
        st.plotly_chart(hm_f, width="stretch")
        st.caption("Entre parênteses: não significativo (|t| < 2). A cor está limitada a |t| = 8 para não esconder "
                   "as ligações moderadas.")

        acc = links[links["aceite"].astype(bool)].copy()
        acc["efeito"] = np.where(acc["sinal"] == "+", "agrava", "atenua")
        acc = acc.reindex(acc["t"].abs().sort_values(ascending=False).index)
        st.dataframe(pd.DataFrame({
            "Fator": acc["fator"].map(lambda f: MACRO_FACTORS[f]["nome"]),
            "Setor": acc["setor"].map(nome),
            "Movimento adverso": acc["efeito"],
            "β": acc["beta"].map(lambda v: fm.num(v, 3, sinal=True)),
            "t (HAC)": acc["t"].map(lambda v: fm.num(v, 1, sinal=True)),
            "R²": acc["r2"].map(lambda v: fm.pct(v, 0)),
            "Semanas": acc["n"],
        }), hide_index=True, width="stretch")
        st.caption(f"{len(acc)} de {len(links)} ligações aceites. β: retorno semanal do setor por +1 p.p. "
                   "(juros, inflação, spreads) ou por +1% (VIX, petróleo, dólar, em log). R²: parte da variação "
                   "semanal do setor explicada pelo fator. São regressões de um fator de cada vez e, com 66 testes, "
                   "cerca de 3 ligações podem ser falsos positivos.")
