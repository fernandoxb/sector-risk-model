"""Modelo: backtest do VaR e do ES nos setores e nas empresas.

Gerado a partir do antigo app.py: o código do separador é o mesmo; os dados do run chegam em `g`.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config.settings import BT_LEVELS, BT_SIGNIF, BT_TL_DAYS, BT_WINDOW, CVAR_LEVEL, OFFICIAL_VAR_METHOD
from src.models import backtest_empresas as be
from src.report import formato as fm
from paineis.comum import GRID, METODOS, MUTED, N_CVAR, RISCO, SEM, layout, metrica, nome, seletor_periodo


def render(g: dict) -> None:
    BT_SER = g["BT_SER"]
    bt_res = g["bt_res"]
    comp_bt = g["comp_bt"]
    comp_risk = g["comp_risk"]
    ler_series = g["ler_series"]
    setores = g["setores"]

    if bt_res is None or not BT_SER.exists():
        st.info("Ainda não há resultados de backtesting. Corre `python run_pipeline.py --offline`.")
    else:
        bt_s = ler_series(str(BT_SER), BT_SER.stat().st_mtime)
        st.subheader("Qual método prevê melhor o risco?")
        st.caption(f"Cada dia, o VaR e o ES são previstos só com dados até à véspera (janela móvel de {BT_WINDOW} "
                   f"dias úteis) e comparados com a perda real. Contagem de setores (de 11) em que o teste "
                   f"**não rejeita** o modelo a {fm.pct(BT_SIGNIF, 0)}.")
        ok = bt_res.assign(Kupiec=bt_res["p_kupiec"] >= BT_SIGNIF, Christoffersen=bt_res["p_christoffersen"] >= BT_SIGNIF,
                           Condicional=bt_res["p_cc"] >= BT_SIGNIF, **{"Z2 (ES)": bt_res["p_z2"] >= BT_SIGNIF})
        resumo = ok.groupby(["metodo", "nivel"])[["Kupiec", "Christoffersen", "Condicional", "Z2 (ES)"]].sum()
        resumo = resumo.reset_index().assign(
            Método=lambda d: d["metodo"].map(METODOS) + np.where(d["metodo"] == OFFICIAL_VAR_METHOD, " ★ oficial", ""),
            Nível=lambda d: d["nivel"].map(fm.nivel))
        resumo = resumo.sort_values(["nivel", "metodo"], ascending=[False, True])
        st.dataframe(resumo[["Método", "Nível", "Kupiec", "Christoffersen", "Condicional", "Z2 (ES)"]],
                     hide_index=True, width="stretch", column_config={
                         c: st.column_config.ProgressColumn(c, min_value=0, max_value=11, format="%d / 11",
                                                            help=h) for c, h in {
                             "Kupiec": "A frequência das exceções é a esperada?",
                             "Christoffersen": "As exceções aparecem espalhadas (e não em grupos)?",
                             "Condicional": "Kupiec e Christoffersen ao mesmo tempo.",
                             "Z2 (ES)": "As perdas nos dias de exceção têm o tamanho que o ES previa?"}.items()})
        st.caption(f"O **{METODOS[OFFICIAL_VAR_METHOD]}** é o único que passa a frequência, a independência e o "
                   "teste do ES ao mesmo tempo, por isso é o modelo oficial. O histórico reage tarde às crises "
                   "(exceções em grupos); o GARCH com t-Student tem a cauda demasiado fina.")

        if comp_bt is not None and comp_risk is not None:
            with st.expander(f"E nas {len(comp_risk)} empresas? O mesmo backtest, empresa a empresa"):
                rs = be.resumo_metodos(comp_bt)
                n_emp = int(rs["series"].max())
                rs = rs.assign(Método=rs["metodo"].map(METODOS), Nível=rs["nivel"].map(fm.nivel)).sort_values(
                    ["nivel", "metodo"], ascending=[False, True])
                st.dataframe(rs[["Método", "Nível", *be.TESTES]], hide_index=True, width="stretch", column_config={
                    c: st.column_config.ProgressColumn(c, min_value=0, max_value=n_emp, format=f"%d / {n_emp}")
                    for c in be.TESTES})
                st.caption(f"Previsões desde 2018 (preços desde 2015). O {METODOS[OFFICIAL_VAR_METHOD]} continua a ser o "
                           f"melhor, mas com menos folga do que nos setores: passa os três testes a {N_CVAR} em "
                           f"{int(comp_risk['validado'].sum())} de {len(comp_risk)} empresas, e em "
                           f"{int(comp_risk['es_subestimado'].sum())} o ES fica subestimado. Uma ação isolada tem saltos "
                           "(resultados, notícias) que um setor dilui. Nessas empresas, o ES mostrado no cockpit é "
                           "multiplicado por k = 1 − Z2 (o fator que teria acertado no backtest), em média "
                           f"×{fm.num(comp_risk.loc[comp_risk['mult_es'] > 1, 'mult_es'].mean())}. O backtest completo das "
                           "empresas corre uma vez por semana; a previsão é refeita todos os dias.")

        st.subheader("Um setor em detalhe")
        b1, b2, b3 = st.columns(3)
        bt_sel = b1.selectbox("Setor", setores, format_func=nome, key="bt_setor")
        met_opts = list(METODOS)
        bt_met = b2.radio("Método", met_opts, index=met_opts.index(OFFICIAL_VAR_METHOD), format_func=METODOS.get,
                          horizontal=True)
        lv_opts = list(BT_LEVELS)
        bt_lv = b3.radio("Nível do VaR", lv_opts, index=lv_opts.index(CVAR_LEVEL) if CVAR_LEVEL in lv_opts else 0,
                         format_func=fm.nivel, horizontal=True)
        rotulo = f"VaR {fm.nivel(bt_lv)} · 1 dia"
        rotulo_es = f"ES {fm.nivel(bt_lv)} · 1 dia"

        s = bt_s[bt_s["setor"] == bt_sel].set_index("data")
        var_col, es_col = f"var_{bt_met}_{bt_lv}", f"es_{bt_met}_{bt_lv}"
        exc = -s["ret"] > s[var_col]
        linha = bt_res[(bt_res["setor"] == bt_sel) & (bt_res["metodo"] == bt_met)
                       & np.isclose(bt_res["nivel"], bt_lv)].iloc[0]

        def veredito(p: float) -> str:
            return "❌ rejeita" if p < BT_SIGNIF else "✅ passa"

        m1, m2, m3, m4, m5 = st.columns(5)
        metrica(m1, "Exceções", fm.num(linha["excecoes"], 0), f"esperadas {fm.num(linha['esperadas'], 0)}",
                ajuda="Dias em que a perda real ultrapassou o VaR previsto.")
        metrica(m2, "Taxa observada", fm.pct(linha["taxa_obs"], 2), f"esperada {fm.pct(linha['taxa_esp'], 1)}")
        metrica(m3, "Kupiec", veredito(linha["p_kupiec"]), f"p = {fm.num(linha['p_kupiec'], 3)}",
                ajuda="Testa se a taxa de exceções é a esperada.")
        metrica(m4, "Z2 do ES", veredito(linha["p_z2"]),
                f"Z2 = {fm.num(linha['z2'], 2, sinal=True)} · p = {fm.num(linha['p_z2'], 3)}",
                ajuda="Acerbi–Székely: Z2 negativo quer dizer ES subestimado.")
        metrica(m5, f"Semáforo ({BT_TL_DAYS} dias)", SEM[linha["semaforo"]],
                f"{int(linha[f'excecoes_{BT_TL_DAYS}'])} exceções · FRTB {linha['frtb_mesa']}",
                ajuda="Zonas de Basileia pelas exceções dos últimos 250 dias (a 99%: verde 0–4, amarelo 5–9, "
                      "vermelho 10+).")

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=s.index, y=s["ret"], mode="lines", name="Retorno diário",
                                    line=dict(width=0.8, color=MUTED)))
        fig_bt.add_trace(go.Scatter(x=s.index, y=-s[var_col], mode="lines", name=f"−{rotulo} previsto",
                                    line=dict(width=1.6, color="#38bdf8")))
        fig_bt.add_trace(go.Scatter(x=s.index, y=-s[es_col], mode="lines", name=f"−{rotulo_es} previsto",
                                    line=dict(width=1, color="#a855f7", dash="dot")))
        fig_bt.add_trace(go.Scatter(x=s.index[exc], y=s["ret"][exc], mode="markers", name="Exceção",
                                    marker=dict(size=5, color=RISCO["vermelho"], line=dict(width=0))))
        layout(fig_bt, 440, hovermode="x unified", margin=dict(l=0, r=0, t=60, b=0),
               legend=dict(orientation="h", y=1.12, yanchor="bottom", x=0),
               yaxis=dict(tickformat=".0%", gridcolor=GRID, zeroline=False), xaxis=dict(gridcolor=GRID))
        seletor_periodo(fig_bt)
        st.plotly_chart(fig_bt, width="stretch")
        st.caption(f"{METODOS[bt_met]}. Exceção = perda real maior do que o VaR previsto na véspera. Abre nos últimos "
                   "3 anos; usa os botões para ver a história toda (desde 2008, com a crise e a COVID-19).")

        with st.expander("Tabela completa (todos os setores, métodos e níveis)"):
            so_setor = st.toggle("Só o setor selecionado", value=False)
            t_bt = bt_res[bt_res["setor"] == bt_sel] if so_setor else bt_res
            st.dataframe(pd.DataFrame({
                "Setor": t_bt["setor"], "Método": t_bt["metodo"].map(METODOS), "Nível": t_bt["nivel"].map(fm.nivel),
                "Dias": t_bt["dias"], "Exceções": t_bt["excecoes"],
                "Taxa": t_bt["taxa_obs"].map(lambda v: fm.pct(v, 2)) + " vs " + t_bt["taxa_esp"].map(fm.pct),
                "Kupiec": t_bt["p_kupiec"].map(veredito), "Christoffersen": t_bt["p_christoffersen"].map(veredito),
                "Condicional": t_bt["p_cc"].map(veredito), "Z2 (ES)": t_bt["p_z2"].map(veredito),
                "p Kupiec": t_bt["p_kupiec"].map(lambda v: fm.num(v, 3)),
                "p Z2": t_bt["p_z2"].map(lambda v: fm.num(v, 3)),
                f"Exceções {BT_TL_DAYS}d": t_bt[f"excecoes_{BT_TL_DAYS}"], "Semáforo": t_bt["semaforo"].map(SEM),
                "FRTB": t_bt["frtb_mesa"],
            }), hide_index=True, width="stretch")
