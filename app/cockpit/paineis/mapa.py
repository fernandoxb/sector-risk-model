"""Teia de risco: fatores macro → setores → empresas (teia.py), o simulador de choques (cenario.py) e, por baixo, o
setor aberto (ficha e empresas), a ficha da empresa ou, sem setor aberto, o score dos setores e os fatores macro.

As ligações entre setores podem ser por quedas conjuntas (dependência de cauda, src/models/cauda.py) ou por correlação.
Os dados do run chegam em `g` (as variáveis do app.py).
"""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config.settings import (BT_SIGNIF, BT_TL_DAYS, CVAR_LEVEL, CVAR_YEARS, MACRO_FACTORS, MARKET, MODEL_RISK_LEVEL,
                             OFFICIAL_VAR_METHOD, PANIC_Q, SECTORS, SECTOR_PROFILE, TRADING_DAYS)
from src.models import carteira as ca, cauda as cd
from src.models import cockpit as ck
from src.models.scorecard import COMPONENTS
from src.report import formato as fm
from src.report.alerts import pi_level
from paineis import cenario, teia
from paineis import fatores as fatores_p, setores as setores_p
from paineis.comum import (CORES_C, GRID, MUTED, NOMES, N_CVAR, OFICIAL, RISCO, SEM, TXT, layout, metrica, modelo_txt, nome,
                           seletor_periodo)


def render(g: dict) -> None:
    BT_SER = g["BT_SER"]
    PIPE_ASOF = g["PIPE_ASOF"]
    PREV_DATA = g["PREV_DATA"]
    bt_res = g["bt_res"]
    comp_risk = g["comp_risk"]
    comp_fat = g["comp_fat"]
    empresa_dados = g["empresa_dados"]
    es_emp = g["es_emp"]
    grafo = g["grafo"]
    cauda = g["cauda"]
    cauda_meta = g["cauda_meta"]
    cenarios = g["cenarios"]
    hold = g["hold"]
    janela = g["janela"]
    ler_series = g["ler_series"]
    links = g["links"]
    metr = g["metr"]
    off = g["off"]
    periodo = g["periodo"]
    precos = g["precos"]
    sc = g["sc"]
    setores = g["setores"]
    stt = g["stt"]

    # --- clique no gráfico (evento do run anterior) → estado, antes de desenhar os seletores
    if "mapa_setor" not in st.session_state:        # 1.ª vez: ligação direta, ex. ?setor=XLK&empresa=NVDA
        q_set = st.query_params.get("setor", "").upper()
        st.session_state["mapa_setor"] = q_set if q_set in SECTORS else None
        st.session_state["mapa_empresa"] = st.query_params.get("empresa", "").upper() or None
    if st.session_state.pop("mapa_reset", False):
        st.session_state["mapa_setor"], st.session_state["mapa_empresa"] = None, None
    Q_TXT = fm.pct(cauda_meta.get("q", 0.05), 0)
    MODOS = {"cauda": "Quedas conjuntas (cauda)", "corr": "Correlação"}
    modo = st.radio("Ligar por", list(MODOS), format_func=MODOS.get, horizontal=True, key="mapa_modo",
                    disabled=cauda is None,
                    help=f"Quedas conjuntas: quando um está nos {Q_TXT} piores dias, com que frequência o outro também "
                         "está. É o que interessa para o contágio em crises. A correlação mede o andamento conjunto "
                         "em dias normais e subestima as quedas conjuntas.") if cauda is not None else "corr"
    por_cauda = modo == "cauda"
    if por_cauda:                               # espinha de contágio pelas quedas conjuntas dos 11 setores
        grafo = cd.graph(cauda[cauda["grupo"] == cd.GRUPO_SETORES], by="p_conj")

    def liga_curta(d: dict) -> str:
        return f"{fm.pct(d['p_conj'], 0)} dos piores dias" if "p_conj" in d else f"r = {fm.num(d['corr'])}"

    def forca(d: dict) -> float:
        return d["p_conj"] if "p_conj" in d else d["corr"]
    if True:                                    # a teia: fatores → setores → empresas (ECharts)
        cen, chocados = (None, [])
        cart = st.session_state.get("cart_df")                 # a carteira da Minha carteira (ou a do link)
        p_cart = (dict(zip(cart["Ativo"], cart["Peso (%)"])) if cart is not None
                  else ca.parse_carteira(st.query_params.get("carteira")))
        p_cart = {t: v for t, v in p_cart.items() if isinstance(t, str) and v == v and v > 0}
        pesos = ca.normalizar(p_cart) if p_cart else None
        if g["cen_sem"] is not None:
            cen, chocados = cenario.render(g, str(g["cen_sem"]["data"].max()), pesos)
        novo = teia.desenhar(g, st.session_state.get("mapa_setor"), Q_TXT, cen, chocados, "p_conj" if por_cauda else "corr",
                             set(p_cart))
        if novo:
            if novo in MACRO_FACTORS:               # fator: mostra (ou esconde) a sua cadeia
                st.session_state["teia_foco"] = None if st.session_state.get("teia_foco") == novo else novo
            elif novo in SECTORS:                   # setor: abre as empresas, como o seletor abaixo
                st.session_state["mapa_setor"], st.session_state["mapa_empresa"] = novo, None
                st.session_state["teia_foco"] = novo
            elif "/" in novo:                       # empresa: abre a ficha
                st.session_state["mapa_setor"], st.session_state["mapa_empresa"] = novo.split("/")
            st.rerun()
        st.caption("**Centro:** fatores macro, ligados aos setores em que mexem mesmo com os outros fatores fixos "
                   "(vermelho cheio agrava, verde tracejado atenua). **Anel do meio:** setores, cor = score, tamanho = ES "
                   f"previsto, ligados pelos que mais caem juntos nos {Q_TXT} piores dias. **Anel de fora:** as 10 maiores "
                   "empresas de cada setor (contorno branco: as que tens na Minha carteira). Clica num fator para ver o choque a "
                   "correr pela cadeia, num setor para abrir as empresas e numa empresa para a ficha; roda do rato para ampliar.")
    m1, m2, m3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    m_set = m1.selectbox("Setor aberto", [None] + setores, key="mapa_setor", placeholder="— todos os setores —",
                         format_func=lambda t: "— todos os setores —" if t is None else nome(t))
    h_set = hold[hold["setor"] == m_set].reset_index(drop=True) if m_set else hold.iloc[:0]
    tick_set = h_set["ticker"].tolist()
    if st.session_state.get("mapa_empresa") not in tick_set:
        st.session_state["mapa_empresa"] = None
    nomes_emp = dict(zip(h_set["ticker"], h_set["nome"]))
    m_emp = m2.selectbox("Empresa", [None] + tick_set, key="mapa_empresa", disabled=not m_set,
                         placeholder="— escolhe uma empresa —",
                         format_func=lambda t: "— escolhe uma empresa —" if t is None else f"{t} · {nomes_emp[t]}")
    if m3.button("⬅ Fechar setor", disabled=not m_set, width="stretch"):
        st.session_state["mapa_reset"] = True
        st.rerun()

    # --- dados das empresas do setor aberto
    r_emp, t_emp, g_emp = None, None, None
    if m_set:
        try:
            r_emp = ck.returns(precos(tuple(tick_set + [m_set, MARKET]), periodo))
            t_emp = ck.tail_table(r_emp[tick_set], r_emp[MARKET], CVAR_LEVEL, PANIC_Q)
            t_emp["corr_etf"] = r_emp[tick_set].corrwith(r_emp[m_set])
            t_emp["vol"] = r_emp[tick_set].std() * np.sqrt(TRADING_DAYS)
            t_emp["peso"] = h_set.set_index("ticker")["peso"]
            cvar_etf = ck.cvar(r_emp[m_set], CVAR_LEVEL)
            t_emp["racio"] = t_emp["cvar"] / cvar_etf
            g_emp, _ = ck.mst(r_emp[tick_set])
            if por_cauda:                       # ligações entre as empresas do setor pelas quedas conjuntas
                pe = cauda[(cauda["grupo"] == m_set) & cauda["a"].isin(tick_set) & cauda["b"].isin(tick_set)]
                if set(pe["a"]) | set(pe["b"]) == set(tick_set):
                    g_emp = cd.graph(pe, by="p_conj")
        except Exception as e:  # noqa: BLE001
            st.warning(f"Sem preços para as empresas de {m_set}: {e}")
            m_set, tick_set = None, []

    if not m_set:
        if por_cauda:
            with st.expander("Todas as ligações: quedas conjuntas contra correlação"):
                ps = cauda[cauda["grupo"] == cd.GRUPO_SETORES]
                n_cauda = int(cauda_meta.get("q", 0.05) * cauda_meta["grupos"][cd.GRUPO_SETORES]["dias"])
                mat = pd.DataFrame(np.nan, index=setores, columns=setores)
                for r_ in ps.itertuples():
                    mat.loc[r_.a, r_.b] = mat.loc[r_.b, r_.a] = r_.p_conj
                hm_c = go.Figure(go.Heatmap(
                    z=mat.values, x=setores, y=setores, zmin=0, zmax=max(0.6, float(np.nanmax(mat.values))),
                    colorscale=[[0, "#131c2e"], [1, RISCO["vermelho"]]], xgap=2, ygap=2,
                    text=[["" if pd.isna(v) else fm.pct(v, 0) for v in row] for row in mat.values],
                    texttemplate="%{text}", colorbar=dict(title="quedas<br>conjuntas", thickness=10)))
                layout(hm_c, 420, f"Quando um setor está nos {Q_TXT} piores dias, o outro também está em…",
                       yaxis=dict(autorange="reversed"))
                st.plotly_chart(hm_c, width="stretch", config={"displayModeBar": False})
                top_c = ps.sort_values("p_conj", ascending=False).head(5)
                dif = ps.assign(posto_c=ps["corr"].rank(ascending=False), posto_q=ps["p_conj"].rank(ascending=False))
                sub = dif[dif["posto_q"] <= 10].sort_values("posto_c", ascending=False).head(3)
                st.caption("Pares que mais caem juntos: " + "; ".join(f"{r_.a}–{r_.b} {fm.pct(r_.p_conj, 0)}" for r_ in top_c.itertuples())
                           + ". Pares que a correlação subestima (estão no top 10 das quedas conjuntas mas mais abaixo na "
                           "correlação): " + "; ".join(f"{r_.a}–{r_.b} (correlação {fm.num(r_.corr)}, {int(r_.posto_c)}.º)"
                                                       for r_ in sub.itertuples())
                           + f". Cópula t ajustada aos resíduos do GARCH: ν = {fm.num(cauda_meta['grupos'][cd.GRUPO_SETORES]['nu'], 1)} "
                           f"({cauda_meta['grupos'][cd.GRUPO_SETORES]['dias']} dias). Com cerca de {n_cauda} dias de cauda por setor, "
                           f"cada percentagem tem uma margem de erro de uns {fm.num(100 * (0.3 * 0.7 / n_cauda) ** 0.5, 0)} pontos.")

    # --- painel: empresas do setor ou ficha da empresa
    if m_set and not m_emp:
        fs = m_set
        st.subheader(nome(fs))
        if sc is not None and off is not None and metr is not None:
            r = sc.loc[fs]
            p_, i_ = int(r["probabilidade"]), int(r["impacto"])
            st.caption(f"Setor {SECTOR_PROFILE[fs]['tipo'].lower()} · resultados do pipeline até {PIPE_ASOF} · "
                       "clica numa empresa do mapa para ver a sua ficha")
            f1, f2, f3, f4, f5 = st.columns(5)
            metrica(f1, "Score de risco", fm.num(r["score_final"], 0), f"{int(r['rank'])}.º de {len(sc)}")
            metrica(f2, "P × I", str(p_ * i_), f"P{p_} × I{i_} · nível {pi_level(p_, i_)}")
            metrica(f3, f"ES {N_CVAR} para {PREV_DATA}", fm.pct(off.loc[fs, "es975"]), f"VaR 99%: {fm.pct(off.loc[fs, 'var99'])}",
                    ajuda=f"Previsão oficial ({OFICIAL}) da perda a 1 dia.")
            metrica(f4, "Semáforo do modelo", SEM[r["semaforo"]],
                    "⚠ score agravado" if r["risco_modelo"] else "sem agravamento",
                    ajuda=f"Backtest do {OFICIAL} a {fm.nivel(MODEL_RISK_LEVEL)} nos últimos {BT_TL_DAYS} dias.")
            metrica(f5, f"Beta ao {MARKET} · {CVAR_YEARS} anos", fm.num(metr.loc[fs, "beta"]),
                    f"vol. no percentil {fm.num(100 * metr.loc[fs, 'vol_pct'], 0)}",
                    ajuda="Percentil da volatilidade dos últimos 63 dias face a toda a história do setor (regime).")

            col_a, col_b = st.columns(2)
            with col_a:
                med = sc[list(COMPONENTS)].median()
                fig_c = go.Figure()
                fig_c.add_trace(go.Bar(y=[NOMES[k] for k in COMPONENTS], x=[r[k] for k in COMPONENTS], orientation="h",
                                       marker_color=[CORES_C[k] for k in COMPONENTS], name=fs,
                                       text=[fm.num(r[k], 2) for k in COMPONENTS], textposition="outside",
                                       hovertemplate="%{y}: %{text}<extra></extra>"))
                fig_c.add_trace(go.Scatter(y=[NOMES[k] for k in COMPONENTS], x=[med[k] for k in COMPONENTS], mode="markers",
                                           name="mediana dos 11", marker=dict(symbol="line-ns-open", size=22,
                                                                              color=TXT, line=dict(width=2))))
                layout(fig_c, 300, "Componentes do score (0 = o mais baixo dos 11, 1 = o mais alto)",
                       xaxis=dict(range=[0, 1.15], gridcolor=GRID), yaxis=dict(autorange="reversed"),
                       legend=dict(orientation="h", y=-0.15))
                st.plotly_chart(fig_c, width="stretch")
            with col_b:
                st.markdown("**Fatores macro que mexem com o setor**")
                if links is not None:
                    lf = links[(links["setor"] == fs) & links["aceite"].astype(bool)]
                    lf = lf.reindex(lf["t"].abs().sort_values(ascending=False).index)
                    if lf.empty:
                        st.caption("Nenhuma ligação significativa.")
                    for _, l in lf.iterrows():
                        ag = l["sinal"] == "+"
                        st.markdown(f"{'🔺' if ag else '🔻'} **{MACRO_FACTORS[l['fator']]['curto']}** a subir "
                                    f"{'agrava' if ag else 'atenua'} (t = {fm.num(l['t'], 1, sinal=True)}, "
                                    f"R² {fm.pct(l['r2'], 0)})"
                                    + ("" if "aceite_multi" not in l or bool(l["aceite_multi"])
                                       else " · *some com os outros fatores*"))
                viz = sorted(((v, grafo[fs][v]) for v in grafo.neighbors(fs)), key=lambda x: -forca(x[1]))
                st.markdown(("**Cai junto com**  \n" if por_cauda else "**Contágio (vizinhos pela correlação)**  \n")
                            + " · ".join(f"{v} ({liga_curta(d_)})" for v, d_ in viz))
                if cenarios is not None and (cenarios["cenario"] == fs).any():
                    cn = cenarios[cenarios["cenario"] == fs].set_index("serie")
                    st.markdown(f"**Se o {fs} tiver um dia mau** (um dos seus {Q_TXT} piores; perda média "
                                f"{fm.pct(cn.loc[fs, 'perda_obs'])})  \n"
                                + "  \n".join(f"{b}: cai junto em {fm.pct(l['p_obs'], 0)} das vezes, perda média "
                                              f"{fm.pct(l['perda_obs'])} · simulação {fm.pct(l['p_cauda'], 0)} e "
                                              f"{fm.pct(l['perda_media'])}"
                                              for b, l in cn.drop(index=fs).nlargest(3, "p_obs").iterrows()))
                    st.caption(f"Sempre na janela de {CVAR_YEARS} anos: primeiro os dias reais (poucos), depois 200 mil dias simulados pela cópula t. "
                               "A simulação costuma dar menos do que o observado.")
                if stt is not None:
                    st.markdown("**Nas crises históricas**")
                    ss = stt[stt["serie"] == fs].set_index("cenario")
                    for cen, l in ss.iterrows():
                        st.markdown(f"- {cen}: **{fm.pct(-l['perda'])}**" + (" (proxy β × SPY)" if l["proxy"] else ""))
            with st.expander(f"Risco previsto pelo {OFICIAL} desde 2008 e backtest do modelo neste setor"):
                if BT_SER.exists():
                    bs = ler_series(str(BT_SER), BT_SER.stat().st_mtime)
                    bs = bs[bs["setor"] == fs].set_index("data")
                    m = OFFICIAL_VAR_METHOD
                    fig_e = go.Figure()
                    fig_e.add_trace(go.Scatter(x=bs.index, y=bs[f"es_{m}_{CVAR_LEVEL}"], mode="lines", name=f"ES {N_CVAR} previsto",
                                               line=dict(color="#a855f7", width=1.4)))
                    fig_e.add_trace(go.Scatter(x=bs.index, y=bs[f"var_{m}_0.99"], mode="lines", name="VaR 99% previsto",
                                               line=dict(color="#38bdf8", width=1.2)))
                    exc_f = -bs["ret"] > bs[f"var_{m}_{CVAR_LEVEL}"]
                    fig_e.add_trace(go.Scatter(x=bs.index[exc_f], y=-bs["ret"][exc_f], mode="markers",
                                               name=f"Perda acima do VaR {N_CVAR}", marker=dict(size=4, color=RISCO["vermelho"])))
                    layout(fig_e, 380, f"Risco previsto pelo {OFICIAL} ao longo do tempo (perda a 1 dia)",
                           hovermode="x unified", margin=dict(l=0, r=0, t=70, b=0),
                           legend=dict(orientation="h", y=-0.12, yanchor="top", x=0),
                           yaxis=dict(tickformat=".0%", gridcolor=GRID), xaxis=dict(gridcolor=GRID))
                    seletor_periodo(fig_e)
                    st.plotly_chart(fig_e, width="stretch")
                    st.caption("Cada ponto é a previsão feita na véspera. O risco previsto sobe logo que a volatilidade sobe, "
                               "por isso os picos coincidem com as crises (2008, 2020, 2022).")
                if bt_res is not None:
                    bf = bt_res[(bt_res["setor"] == fs) & (bt_res["metodo"] == OFFICIAL_VAR_METHOD)]
                    st.markdown("**Backtest do modelo oficial neste setor**")
                    st.dataframe(pd.DataFrame({
                        "Nível": bf["nivel"].map(fm.nivel),
                        "Exceções": bf["excecoes"].astype(str) + " (esperadas " + bf["esperadas"].round(0).astype(int).astype(str) + ")",
                        "Kupiec": bf["p_kupiec"].map(lambda v: "❌ rejeita" if v < BT_SIGNIF else "✅ passa"),
                        "Christoffersen": bf["p_christoffersen"].map(lambda v: "❌ rejeita" if v < BT_SIGNIF else "✅ passa"),
                        "Z2 (ES)": bf["p_z2"].map(lambda v: "❌ rejeita" if v < BT_SIGNIF else "✅ passa"),
                        f"Semáforo {BT_TL_DAYS}d": bf["semaforo"].map(SEM),
                    }), hide_index=True, width="stretch")
        st.subheader(f"Empresas do {nome(m_set)}")
        ord_e = t_emp.sort_values("peso", ascending=False)
        st.dataframe(pd.DataFrame({
            "Empresa": [f"{t} · {nomes_emp[t]}" for t in ord_e.index],
            "Peso no ETF": ord_e["peso"].map(fm.pct).values,
            f"CVaR {N_CVAR}": ord_e["cvar"].map(lambda v: fm.pct(v, 2)).values,
            "Face ao ETF": ord_e["racio"].map(fm.mult).values,
            f"ES {N_CVAR} previsto": [fm.pct(es_emp(t)) if es_emp(t) is not None else "—" for t in ord_e.index],
            "Modelo": [modelo_txt(comp_risk.loc[t] if comp_risk is not None and t in comp_risk.index else None)
                       for t in ord_e.index],
            "Risco próprio": [fm.pct(comp_fat.loc[t, "pct_especifico"], 0) if comp_fat is not None and t in comp_fat.index
                              else "—" for t in ord_e.index],
            "Downside beta": ord_e["downside_beta"].map(fm.num).values,
            "Correlação com o ETF": ord_e["corr_etf"].map(fm.num).values,
            "Volatilidade anual": ord_e["vol"].map(fm.pct).values,
        }), hide_index=True, width="stretch")
        st.caption(f"CVaR {N_CVAR} · 1 dia · {janela.lower()}, histórico (CVaR do {m_set}: {fm.pct(cvar_etf, 2)}). ES previsto: "
                   f"modelo oficial ({OFICIAL}) para {PREV_DATA}. Modelo: ✅ se passa os três testes do backtest a {N_CVAR} "
                   "(frequência, independência e tamanho das perdas).")
    elif m_set and m_emp:
        dados, origem = empresa_dados(m_emp)
        info, noticias = dados.get("info", {}), dados.get("noticias", [])
        e = t_emp.loc[m_emp]
        with st.container(border=True):
            st.subheader(f"{info.get('nome', nomes_emp[m_emp])} ({m_emp})")
            st.caption(" · ".join(x for x in [info.get("industria"), info.get("pais"),
                                              f"[site]({info['site']})" if info.get("site") else None,
                                              f"dados de bolsa: {origem}"] if x))
            b1, b2, b3, b4, b5 = st.columns(5)
            moeda = info.get("moeda", "USD")
            if "preco" in info:
                var = info.get("var_dia_pct")
                b1.metric("Preço", f"{fm.num(info['preco'])} {moeda}",
                          f"{fm.num(var, 2, sinal=True)}% no dia" if var is not None else None)
            else:
                metrica(b1, "Preço", "—")
            metrica(b2, "Capitalização", fm.grande(info["cap_bolsa"]) if "cap_bolsa" in info else "—", moeda)
            metrica(b3, "PER", fm.num(info["per"], 1) if "per" in info else "—",
                    f"futuro {fm.num(info['per_futuro'], 1)}" if "per_futuro" in info else None,
                    ajuda="Preço a dividir pelo lucro por ação dos últimos 12 meses; 'futuro' usa o lucro estimado.")
            metrica(b4, "Intervalo de 52 semanas",
                    f"{fm.num(info['min_52s'], 0)} – {fm.num(info['max_52s'], 0)}" if "min_52s" in info else "—",
                    f"dividendo {fm.num(info['dividendo_pct'], 2)}%" if info.get("dividendo_pct") else "sem dividendo")
            REC = {"strong_buy": "compra forte", "buy": "comprar", "hold": "manter", "underperform": "abaixo do mercado",
                   "sell": "vender", "strong_sell": "venda forte", "none": "—"}
            metrica(b5, "Analistas", REC.get(str(info.get("recomendacao")), str(info.get("recomendacao", "—"))),
                    f"preço-alvo {fm.num(info['preco_alvo'], 0)}" if "preco_alvo" in info else None,
                    ajuda="Recomendação média dos analistas, segundo o Yahoo Finance. Não é aconselhamento.")
            k1, k2, k3, k4, k5 = st.columns(5)
            metrica(k1, f"Peso no {m_set}", fm.pct(e["peso"]))
            metrica(k2, f"CVaR {N_CVAR} · 1 dia", fm.pct(e["cvar"], 2), f"{fm.mult(e['racio'])} o do {m_set}",
                    ajuda="Perda média nos 2,5% piores dias da janela. Histórico, não backtestado.")
            metrica(k3, "Downside beta", fm.num(e["downside_beta"]), f"beta {fm.num(e['beta'])}",
                    ajuda=f"Sensibilidade ao {MARKET} nos dias de pânico; beta normal ao lado.")
            metrica(k4, f"Correlação com o {m_set}", fm.num(e["corr_etf"]))
            metrica(k5, "Volatilidade anual", fm.pct(e["vol"]))

            if comp_fat is not None and m_emp in comp_fat.index:
                cf = comp_fat.loc[m_emp]
                st.caption(f"**De onde vem o risco da {m_emp}:** {fm.pct(cf['pct_mercado'], 0)} do mercado ({MARKET}), "
                           f"{fm.pct(cf['pct_setor'], 0)} do setor ({m_set}, a parte que o mercado não explica) e "
                           f"{fm.pct(cf['pct_especifico'], 0)} próprio da empresa. Partes da variância diária em "
                           f"{CVAR_YEARS} anos; só o risco próprio se dilui ao juntar empresas.")
            cr = comp_risk.loc[m_emp] if comp_risk is not None and m_emp in comp_risk.index else None
            if cr is None:
                st.caption("Sem modelo de previsão: a empresa ainda não tem histórico suficiente (cerca de 3 anos e "
                           "meio). Fica só com o risco histórico acima.")
            else:
                f1_, f2_, f3_, f4_ = st.columns(4)
                metrica(f1_, f"ES {N_CVAR} previsto · {cr['data_previsao']}", fm.pct(cr["es975_aj"]),
                        f"modelo {fm.pct(cr['es975'])} × {fm.num(cr['mult_es'])}" if cr["mult_es"] > 1
                        else f"histórico: {fm.pct(e['cvar'], 2)}",
                        ajuda=f"Perda média esperada nos piores dias de amanhã, pelo modelo oficial ({OFICIAL}): reage à "
                              "volatilidade atual. O histórico é a média dos piores dias da janela.")
                metrica(f2_, "VaR 99% previsto", fm.pct(cr["var99"]), f"VaR {N_CVAR}: {fm.pct(cr['var975'])}")
                metrica(f3_, f"Semáforo ({BT_TL_DAYS} dias)", SEM[cr["semaforo"]], "exceções ao VaR 99%")
                metrica(f4_, "Backtest do modelo", modelo_txt(cr),
                        "previsão fiável" if cr["validado"] else "ler a previsão com cautela",
                        ajuda=f"Validado: desde 2018, o {OFICIAL} passa nesta empresa os testes de Kupiec, Christoffersen "
                              f"e Z2 a {N_CVAR}. ES corrigido: no backtest as perdas nos dias maus foram maiores do que o "
                              "previsto (típico de ações com saltos em dias de resultados), por isso o ES mostrado já "
                              "vem multiplicado pelo fator que teria acertado no passado.")
            col_p, col_n = st.columns([3, 2])
            with col_p:
                px_ = precos(tuple(tick_set + [m_set, MARKET]), periodo)[[m_emp, m_set]].dropna()
                px_ = px_[px_.index > px_.index.max() - pd.DateOffset(years=1)]
                base = 100 * px_ / px_.iloc[0]
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(x=base.index, y=base[m_emp], mode="lines", name=m_emp,
                                           line=dict(color="#38bdf8", width=2)))
                fig_p.add_trace(go.Scatter(x=base.index, y=base[m_set], mode="lines", name=m_set,
                                           line=dict(color=MUTED, width=1.4, dash="dot")))
                layout(fig_p, 300, "Último ano (base 100)", hovermode="x unified",
                       legend=dict(orientation="h", y=-0.15), yaxis=dict(gridcolor=GRID), xaxis=dict(gridcolor=GRID))
                st.plotly_chart(fig_p, width="stretch", config={"displayModeBar": False})
                viz_e = sorted(((v, g_emp[m_emp][v]) for v in g_emp.neighbors(m_emp)), key=lambda x: -forca(x[1]))
                st.markdown(("**Cai junto com:** " if por_cauda else "**Move-se mais com:** ")
                            + " · ".join(f"{v} ({liga_curta(d_)})" for v, d_ in viz_e))
                if info.get("descricao"):
                    with st.expander("Sobre a empresa (em inglês, do Yahoo Finance)"):
                        st.write(info["descricao"])
            with col_n:
                st.markdown("**Últimas notícias**")
                if not noticias:
                    st.caption("Sem notícias disponíveis.")
                for n_ in noticias:
                    st.markdown(f"[{html.escape(str(n_['titulo']))}]({n_['link']})  \n"
                                f"<span style='color:{MUTED};font-size:0.85em'>{html.escape(str(n_.get('fonte') or ''))} · "
                                f"{fm.data_curta(n_.get('data'))}</span>", unsafe_allow_html=True)
                st.caption(f"Fonte: Yahoo Finance ({origem}). Títulos em inglês; as horas são UTC.")
    else:                                       # sem setor aberto: o score dos setores e os fatores, por baixo da teia
        VISTAS = {"setores": "🎯 Score dos 11 setores", "fatores": "🔗 Todas as ligações aos fatores macro"}
        vista = st.segmented_control("Por baixo da teia", list(VISTAS), format_func=VISTAS.get, default="setores",
                                     key="teia_vista") or "setores"
        (setores_p if vista == "setores" else fatores_p).render(g)
