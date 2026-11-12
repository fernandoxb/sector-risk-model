"""Carteira-base: os 11 setores em carteira, contribuições de Euler, stress históricos e próprio.

Gerado a partir do antigo app.py: o código do separador é o mesmo; os dados do run chegam em `g`.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config.settings import (CVAR_YEARS, LINK_T_MIN, MACRO_FACTORS, MARKET, PORTFOLIO_WEIGHTS, SHOCK_DIFF, SHOCK_LOGDIFF,
                             SHOCK_MARKET)
from src.models.portfolio import shock_returns
from src.report import formato as fm
from paineis.comum import GRID, MUTED, N_CVAR, RISCO, TXT, layout, metrica


def render(g: dict) -> None:
    links = g["links"]
    lw = g["lw"]
    metr = g["metr"]
    rc = g["rc"]
    setores = g["setores"]
    stt = g["stt"]

    if rc is None or stt is None or lw is None:
        st.info("Ainda não há resultados da carteira. Corre `python run_pipeline.py --offline`.")
    else:
        cart, rc_s = rc.loc["Carteira"], rc.drop(index="Carteira")
        rot_cvar = f"CVaR {N_CVAR} · 1 dia · {CVAR_YEARS} anos"
        pesos_txt = "pesos iguais (1/11)" if PORTFOLIO_WEIGHTS is None else "pesos de PORTFOLIO_WEIGHTS"
        st.subheader(f"Carteira dos 11 setores com {pesos_txt}")
        div_vol = (rc_s["peso"] * rc_s["vol_individual"]).sum() / cart["contrib_vol"]
        div_cvar = (rc_s["peso"] * rc_s["cvar_individual"]).sum() / cart["contrib_cvar"]
        k1, k2, k3, k4 = st.columns(4)
        metrica(k1, f"Volatilidade anual · {CVAR_YEARS} anos", fm.pct(cart["contrib_vol"]),
                f"diversificação {fm.mult(div_vol)}",
                ajuda="Diversificação = soma dos riscos isolados ÷ risco da carteira. 1× = nenhuma.")
        metrica(k2, rot_cvar, fm.pct(cart["contrib_cvar"], 2), f"diversificação {fm.mult(div_cvar)}",
                ajuda="Nos piores dias os setores caem juntos, por isso a diversificação é menor na cauda.")
        top_rc = rc_s["pct_cvar"].idxmax()
        metrica(k3, "Maior contribuição para o CVaR", top_rc, f"{fm.pct(rc_s.loc[top_rc, 'pct_cvar'])} do total "
                f"(peso {fm.pct(rc_s.loc[top_rc, 'peso'])})")
        pior = stt[stt["serie"] == "Carteira"].sort_values("perda").iloc[-1]
        metrica(k4, "Pior cenário de stress", fm.pct(-pior["perda"]), pior["cenario"])

        col_rc, col_st = st.columns(2)
        with col_rc:
            fig_rc = go.Figure()
            fig_rc.add_trace(go.Bar(x=rc_s.index, y=rc_s["pct_vol"], name="% da volatilidade", marker_color="#64748b"))
            fig_rc.add_trace(go.Bar(x=rc_s.index, y=rc_s["pct_cvar"], name="% do CVaR", marker_color="#6366f1"))
            fig_rc.add_trace(go.Scatter(x=rc_s.index, y=rc_s["peso"], mode="lines", name="Peso na carteira",
                                        line=dict(color=TXT, dash="dot", width=1)))
            layout(fig_rc, 380, "Quem pesa no risco da carteira (contribuições de Euler)", barmode="group",
                   legend=dict(orientation="h", y=-0.12), yaxis=dict(tickformat=".0%", gridcolor=GRID))
            st.plotly_chart(fig_rc, width="stretch")
        with col_st:
            ordem = setores + ["Carteira", MARKET]
            piv = stt.pivot(index="serie", columns="cenario", values="perda").reindex(ordem)
            prx = stt.pivot(index="serie", columns="cenario", values="proxy").reindex(ordem).fillna(False)
            ret = -piv                                  # retorno acumulado (perda com sinal trocado)
            txt = [[fm.pct(v, 0) + ("*" if p else "") for v, p in zip(rv, rp)] for rv, rp in zip(ret.values, prx.values)]
            hm_st = go.Figure(go.Heatmap(z=ret.values, x=[c.split(" · ")[0] + "<br>" + c.split(" · ")[-1]
                                                          for c in ret.columns], y=ordem, zmid=0, colorscale="RdYlGn",
                                         text=txt, texttemplate="%{text}", showscale=False, xgap=2, ygap=2,
                                         hovertemplate="%{y} · %{x}: %{text}<extra></extra>"))
            layout(hm_st, 380, "Retorno em cada crise (do pico ao vale do SPY)", yaxis=dict(autorange="reversed"))
            st.plotly_chart(hm_st, width="stretch")
        st.caption("A carteira compra no início de cada crise e não re-equilibra (buy-and-hold). As contribuições de "
                   "Euler somam exatamente a volatilidade (covariância Ledoit-Wolf) e o CVaR da carteira. "
                   "* = setor ainda sem cotação no episódio: aproximado por β × SPY com o β atual.")

        cart_st = stt[stt["serie"] == "Carteira"].set_index("cenario")
        contrib = stt[~stt["serie"].isin(["Carteira", MARKET])]
        maior = contrib.loc[contrib.groupby("cenario")["contrib"].idxmax()].set_index("cenario")
        spy_st = stt[stt["serie"] == MARKET].set_index("cenario")
        st.dataframe(pd.DataFrame({
            "Período": cart_st["inicio"] + " → " + cart_st["fim"],
            "Retorno da carteira": (-cart_st["perda"]).map(fm.pct),
            f"Retorno do {MARKET}": (-spy_st["perda"]).map(fm.pct),
            "Drawdown máximo": (-cart_st["max_dd"]).map(fm.pct),
            "Pior dia": (-cart_st["pior_dia"]).map(lambda v: fm.pct(v, 2)),
            "Pior dia em CVaR atuais": cart_st["pior_dia_em_cvar"].map(lambda v: fm.mult(v, 1)),
            "Maior contribuinte para a perda": maior["serie"] + " (" + maior["contrib"].map(fm.pct) + ")",
        }).rename_axis("Cenário").reset_index(), hide_index=True, width="stretch")
        st.caption("Pior dia em CVaR atuais: quantas vezes a pior queda diária do episódio é maior do que o CVaR de "
                   "hoje. Mostra que o CVaR mede dias maus normais, não crises.")

        st.subheader("🧨 Cenário de stress próprio")
        st.caption("Define um choque instantâneo e vê o efeito em cada setor e na carteira, propagado pelos betas medidos. "
                   "Um choque de cada vez (os fatores estão correlacionados entre si, somá-los contaria o mesmo risco "
                   "duas vezes).")
        opcoes = ["mercado"] + list(MACRO_FACTORS)
        rot_opc = {"mercado": f"Mercado ({MARKET})", **{f: MACRO_FACTORS[f]["curto"] for f in MACRO_FACTORS}}
        cc1, cc2 = st.columns([1, 2])
        tipo_ch = cc1.selectbox("Choque em", opcoes, format_func=rot_opc.get)
        if tipo_ch == "mercado":
            lo, hi, dflt, stp = SHOCK_MARKET
            mag = cc2.slider(f"Variação do {MARKET} (%)", lo, hi, dflt, stp, format="%.0f%%")
            betas_ch, choque_x = metr["beta"].reindex(setores), mag / 100
            r_ch = shock_returns(betas_ch, choque_x)
            fonte_b = f"beta diário de {CVAR_YEARS} anos face ao {MARKET}"
            desc = f"{MARKET} {fm.num(mag, 0, sinal=True)}%"
        else:
            meta = MACRO_FACTORS[tipo_ch]
            em_pp = meta["transf"] == "diff"
            lo, hi, dflt, stp = SHOCK_DIFF if em_pp else SHOCK_LOGDIFF
            mag = cc2.slider(f"{meta['nome']}: variação", lo, hi, dflt, stp,
                             format="%+.2f p.p." if em_pp else "%+.0f%%")
            Lf = links[(links["fator"] == tipo_ch)].set_index("setor")
            betas_ch = Lf["beta"].where(Lf["aceite"].astype(bool)).reindex(setores)
            choque_x = mag if em_pp else mag / 100
            r_ch = shock_returns(betas_ch, choque_x, log_model=True, log_shock=not em_pp)
            fonte_b = f"β semanal da regressão HAC (só ligações com |t| ≥ {fm.num(LINK_T_MIN, 0)}; as restantes contam 0)"
            desc = f"{meta['curto']} {fm.num(mag, 2 if em_pp else 0, sinal=True)}{' p.p.' if em_pp else '%'}"
        w_c = rc_s["peso"].reindex(setores)
        r_cart = float((w_c * r_ch).sum())
        x1, x2, x3 = st.columns(3)
        metrica(x1, f"Carteira · {desc}", fm.pct(r_cart, 2), f"{fm.mult(abs(r_cart) / cart['contrib_cvar'], 1)} o CVaR "
                "diário da carteira")
        metrica(x2, "Setor mais atingido", r_ch.idxmin(), fm.pct(r_ch.min(), 2))
        metrica(x3, "Setor mais beneficiado", r_ch.idxmax(), fm.pct(r_ch.max(), 2))
        ordem_ch = r_ch.sort_values()
        fig_ch = go.Figure(go.Bar(x=ordem_ch.index, y=ordem_ch.values,
                                  marker_color=[RISCO["vermelho"] if v < 0 else RISCO["verde"] for v in ordem_ch.values],
                                  text=[fm.pct(v, 1) for v in ordem_ch.values], textposition="outside",
                                  hovertemplate="%{x}: %{text}<extra></extra>"))
        fig_ch.add_hline(y=r_cart, line=dict(color=TXT, dash="dot", width=1),
                         annotation_text=f"carteira {fm.pct(r_cart, 2)}", annotation_font_color=TXT)
        layout(fig_ch, 340, yaxis=dict(tickformat=".0%", gridcolor=GRID, zeroline=True, zerolinecolor=MUTED))
        st.plotly_chart(fig_ch, width="stretch")
        st.caption(f"Retorno de cada setor = β × choque, com o {fonte_b}. É uma aproximação linear: choques grandes "
                   "saem da amostra em que os betas foram estimados e ignoram o contágio que só aparece em crises "
                   "(para isso, ver os cenários históricos acima).")

        with st.expander("📐 Covariância: Ledoit-Wolf face à amostral"):
            st.dataframe(pd.DataFrame({
                "Janela (dias)": lw["janela"],
                "T / N": lw["t_sobre_n"].map(lambda v: fm.num(v, 1)),
                "Encolhimento δ": lw["delta"].map(lambda v: fm.num(v, 3)),
                "Nº de condição amostral": lw["cond_amostral"].map(lambda v: fm.num(v, 1)),
                "Nº de condição Ledoit-Wolf": lw["cond_ledoit_wolf"].map(lambda v: fm.num(v, 1)),
                "Vol. fora da amostra (amostral)": lw["vol_oos_amostral"].map(lambda v: fm.pct(v, 2)),
                "Vol. fora da amostra (LW)": lw["vol_oos_ledoit_wolf"].map(lambda v: fm.pct(v, 2)),
                "Redução": lw["reducao_vol"].map(fm.pct),
                "Desde": lw["oos_inicio"],
            }), hide_index=True, width="stretch")
            st.caption("Carteira de variância mínima re-estimada todos os meses só com dados passados. Quanto menor "
                       "T/N (dias por ativo), mais o encolhimento ajuda.")
