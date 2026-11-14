"""Setores: scorecard de risco 0–100 e evolução do score.

Gerado a partir do antigo app.py: o código do separador é o mesmo; os dados do run chegam em `g`.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config.settings import (BT_TL_DAYS, MODEL_RISK_LEVEL, MODEL_RISK_MULT, MOMENTUM_DAYS, PI_LEVELS, SCORE_WEIGHTS)
from src.models.scorecard import COMPONENTS
from src.report import formato as fm
from src.report.alerts import pi_level
from paineis.comum import CORES_C, CRIT, CRIT_MIN, GRID, NOMES, N_CVAR, OFICIAL, SEM, layout, metrica


def render(g: dict) -> None:
    PIPE_ASOF = g["PIPE_ASOF"]
    hist = g["hist"]
    sc = g["sc"]

    if sc is None:
        st.info("Ainda não há scorecard. Corre `python run_pipeline.py --offline`.")
    else:
        st.subheader("Scorecard de risco por setor (0–100)")
        st.caption("Junta cinco medidas de risco num número: 100 = o setor mais arriscado em todas, 0 = o menos "
                   f"arriscado em todas. Dados até {PIPE_ASOF}.")
        s1, s2, s3 = st.columns(3)
        g1 = sc[sc["grupo"] == sc["grupo"].min()]
        metrica(s1, "Grupo de maior risco", ", ".join(g1.index),
                f"score {fm.num(g1['score_final'].max(), 0)}/100" if len(g1) == 1 else
                f"scores {fm.num(g1['score_final'].min(), 0)}–{fm.num(g1['score_final'].max(), 0)}",
                ajuda="Os setores do mesmo grupo não se distinguem: os seus intervalos de posição sobrepõem-se quando os "
                      "pesos do score variam à volta dos oficiais.")
        crit = sc[sc["impacto"] * sc["probabilidade"] >= CRIT_MIN]
        metrica(s2, f"Setores em nível {CRIT}", str(len(crit)), ", ".join(crit.index) or "nenhum",
                ajuda=f"Nível {CRIT}: probabilidade × impacto (P × I) ≥ {CRIT_MIN}, numa escala de 1 a 25. "
                      "A mesma regra dos alertas do Sentinela.")
        flag = sc[sc["risco_modelo"].astype(bool)]
        metrica(s3, "Setores com risco de modelo", str(len(flag)), ", ".join(flag.index) or "nenhum",
                ajuda=f"O backtest do {OFICIAL} a {fm.nivel(MODEL_RISK_LEVEL)} não está verde nos últimos "
                      f"{BT_TL_DAYS} dias: o score é agravado ({', '.join(f'{c} ×{fm.num(m)}' for c, m in MODEL_RISK_MULT.items())}).")

        col_g, col_t = st.columns([5, 6])
        with col_g:
            ordem = sc.index[::-1]
            fig_sc = go.Figure()
            for k in COMPONENTS:
                fig_sc.add_trace(go.Bar(y=ordem, x=100 * SCORE_WEIGHTS[k] * sc.loc[ordem, k], orientation="h",
                                        name=f"{NOMES[k]} ({fm.pct(SCORE_WEIGHTS[k], 0)})", marker_color=CORES_C[k],
                                        hovertemplate="%{y}: %{x:.1f} pontos<extra>" + NOMES[k] + "</extra>"))
            extra = (sc.loc[ordem, "score_final"] - sc.loc[ordem, "score"]).clip(lower=0)
            fig_sc.add_trace(go.Bar(y=ordem, x=extra, orientation="h", name="Risco de modelo",
                                    marker=dict(color="rgba(234,179,8,0.35)", pattern_shape="/",
                                                line=dict(color="#eab308", width=1))))
            layout(fig_sc, 470, "Pontos de cada componente", barmode="stack",
                   xaxis=dict(range=[0, 100], gridcolor=GRID), margin=dict(l=0, r=0, t=90, b=0),
                   legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=11),
                               entrywidth=0.33, entrywidthmode="fraction", traceorder="normal"))
            st.plotly_chart(fig_sc, width="stretch")
        with col_t:
            st.dataframe(pd.DataFrame({
                "Grupo": sc["grupo"].astype(int).values,
                "Setor": list(sc.index),
                "Score": sc["score_final"].round(0).astype(int),
                **({"Posição": [f"{a}–{b}" if a != b else f"{a}" for a, b in zip(sc["rank_p05"], sc["rank_p95"])]}
                   if "rank_p05" in sc else {}),
                "P × I": [f"{p * i} · {pi_level(p, i)}" for p, i in zip(sc["probabilidade"], sc["impacto"])],
                "Modelo": sc["semaforo"].map(lambda v: SEM[v].split()[0]) + sc["risco_modelo"].map({True: " ⚠", False: ""}),
                "ES amanhã": sc["bruto_cauda"].map(fm.pct),
            }), hide_index=True, width="stretch", height=470, column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                "Grupo": st.column_config.NumberColumn("Grupo", help="Setores do mesmo grupo não se distinguem: os "
                                                       "intervalos de posição sobrepõem-se.", width="small"),
                "Posição": st.column_config.TextColumn("Posição", help="Entre que posições o setor fica em 90% dos casos "
                                                       "quando os pesos do score variam à volta dos oficiais."),
                "P × I": st.column_config.TextColumn("P × I", help="Probabilidade (1–5) × impacto (1–5) e nível."),
                "Modelo": st.column_config.TextColumn("Modelo", help="Semáforo do backtest oficial; ⚠ = risco de modelo."),
                "ES amanhã": st.column_config.TextColumn(f"ES {N_CVAR}", help=f"ES {N_CVAR} previsto pelo {OFICIAL}."),
            })
        with st.expander("Como se calcula o score"):
            st.markdown(
                "Cada componente é ordenado entre os 11 setores e vai de 0 (o mais baixo) a 1 (o mais alto); "
                "o score é 100 × a soma ponderada. Na cauda, metade vem dessa ordem e metade de comparar o ES de hoje "
                "com a história do próprio setor, para um score baixo não ser só \"o menos arriscado dos 11\".\n\n"
                f"| Componente | Peso | O que mede |\n|---|---|---|\n"
                f"| Cauda | {fm.pct(SCORE_WEIGHTS['cauda'], 0)} | ES {N_CVAR} previsto pelo {OFICIAL} para amanhã |\n"
                f"| Sistémico | {fm.pct(SCORE_WEIGHTS['sistemico'], 0)} | quanto o setor pesa no CVaR da carteira, por unidade de peso (Euler) |\n"
                f"| Stress | {fm.pct(SCORE_WEIGHTS['stress'], 0)} | perda média nos cenários 2008, COVID-19 e 2022 |\n"
                f"| Regime | {fm.pct(SCORE_WEIGHTS['regime'], 0)} | volatilidade dos últimos 63 dias face à história do próprio setor |\n"
                f"| Momento | {fm.pct(SCORE_WEIGHTS['momento'], 0)} | exceções ao VaR {N_CVAR} oficial nos últimos {MOMENTUM_DAYS} dias |\n\n"
                "**P × I**: impacto (I) = quintil de cauda + sistémico + stress; probabilidade (P) = quintil de "
                "regime + momento. Níveis: "
                + ", ".join(f"{n} ≤ {lim}" for lim, n in PI_LEVELS) + ". Detalhes na nota metodológica, secção 8.")

        st.subheader("Evolução do score")
        if hist is None or hist["data"].nunique() < 2:
            st.caption("O histórico do score começa a acumular-se com os runs diários do pipeline (bot Sentinela).")
        else:
            fig_h = go.Figure()
            for t in sc.index:
                h = hist[hist["setor"] == t]
                fig_h.add_trace(go.Scatter(x=pd.to_datetime(h["data"]), y=h["score"], mode="lines+markers", name=t,
                                           hovertemplate=t + " · %{x|%d/%m}: %{y:.1f}<extra></extra>"))
            layout(fig_h, 320, yaxis=dict(range=[0, 100], gridcolor=GRID, title="Score"),
                   xaxis=dict(gridcolor=GRID, tickformat="%d/%m"), legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_h, width="stretch")
            st.caption(f"{hist['data'].nunique()} dias com estado gravado ({hist['data'].min()} → {hist['data'].max()}). "
                       "Um ponto por run diário do pipeline.")
