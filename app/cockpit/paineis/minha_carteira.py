"""Minha carteira: carteira do utilizador, risco histórico e previsto, mapa, posições e stress.

Os dados do run chegam em `g` (as variáveis do app.py).
"""
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config.settings import (BT_SIGNIF, BT_TL_DAYS, CVAR_LEVEL, MARKET, PANIC_Q, SECTORS, SHOCK_MARKET, USER_FORECAST_YEARS,
                             USER_PORTFOLIO_DEFAULT, USER_PORTFOLIO_MAX)
from config.settings import COMPANY_TTL, HRP_MIN_DAYS, USER_MARKET_LABEL
from src.collect.empresas import validar_ticker
from src.models import carteira as ca, cauda as cd, cockpit as ck
from src.models.portfolio import shock_returns
from src.report import formato as fm
from src.report.alerts import pi_level
from paineis.comum import (CRIT, CRIT_MIN, GRID, MUTED, N_CVAR, OFICIAL, RISCO, SEM, TXT, layout, metrica, modelo_txt, nome)


@st.cache_data(ttl=COMPANY_TTL, show_spinner="🔎 A procurar a ação no Yahoo…")
def _validar(texto: str) -> dict | None:
    return validar_ticker(texto)


def tickers_do_link(link: str) -> list[str]:
    """Tickers escritos no link (?carteira=AAPL:30,ASML.AS:10), em maiúsculas, pela ordem."""
    return [p.partition(":")[0].strip().upper() for p in link.replace(";", ",").split(",") if ":" in p]


def render(g: dict) -> None:
    alertas = g["alertas"]
    backtest_carteira = g["backtest_carteira"]
    es_emp = g["es_emp"]
    grafo = g["grafo"]
    cauda = g["cauda"]
    if cauda is not None:                    # espinha de contágio pelas quedas conjuntas, como na Teia de risco
        grafo = cd.graph(cauda[cauda["grupo"] == cd.GRUPO_SETORES], by="p_conj")
    hold = g["hold"]
    janela = g["janela"]
    off = g["off"]
    periodo = g["periodo"]
    precos = g["precos"]
    sc = g["sc"]
    setores = g["setores"]
    stt = g["stt"]

    extras = st.session_state.setdefault("extras", {})     # ações fora do mapa: ticker → {nome, setor}
    base = list(dict.fromkeys(hold["ticker"])) + setores + [MARKET]
    if "cart_df" not in st.session_state:                   # 1.ª vez: as ações do link que não estão no mapa
        for t_ in tickers_do_link(st.query_params.get("carteira") or ""):
            if t_ not in base and t_ not in extras and (ficha := _validar(t_)):
                extras[ficha["ticker"]] = ficha
    SET_DE = ca.mapa_setores(hold) | {t: (x["setor"] or USER_MARKET_LABEL) for t, x in extras.items()}
    UNIVERSO = base + [t for t in extras if t not in base]
    NOME_AT = (dict(zip(hold["ticker"], hold["nome"])) | {s_: f"ETF {SECTORS[s_]}" for s_ in setores}
               | {MARKET: "ETF S&P 500"} | {t: f"{x['nome']} (fora do mapa)" for t, x in extras.items()})

    def rot_ativo(t: str) -> str:
        return f"{t} · {NOME_AT.get(t, t)}"

    if "cart_df" not in st.session_state:           # 1.ª vez: carteira do link (?carteira=AAPL:30,...) ou exemplo
        link = st.query_params.get("carteira") or ""
        ini = ca.parse_carteira(link, set(UNIVERSO))
        if ini and len([p for p in link.replace(";", ",").split(",") if p.strip()]) > len(ini):
            st.warning("Algumas posições do link foram ignoradas. O formato é TICKER:peso, separado por vírgulas, "
                       "com ponto nas décimas (AAPL:12.5).")
        if not ini:
            ini = {hold[hold["setor"] == s_].iloc[0]["ticker"]: 100 / USER_PORTFOLIO_DEFAULT
                   for s_ in setores[:USER_PORTFOLIO_DEFAULT]}
        st.session_state["cart_df"] = pd.DataFrame({"Ativo": list(ini), "Peso (%)": [round(v, 1) for v in ini.values()]})
        st.session_state["cart_v"] = 0

    st.caption(f"Monta uma carteira com as {hold['ticker'].nunique()} empresas do mapa, os 11 ETFs setoriais, o SPY ou "
               f"qualquer outra ação (até {USER_PORTFOLIO_MAX} posições) e vê o risco, a exposição por setor e as ligações de contágio. "
               "Guarda-se no link, sem conta.")
    col_ed, col_bt = st.columns([3, 1])
    with col_ed:
        ed = st.data_editor(
            st.session_state["cart_df"], num_rows="dynamic", hide_index=True, width="stretch",
            key=f"cart_editor_{st.session_state['cart_v']}",
            column_config={
                "Ativo": st.column_config.SelectboxColumn("Ativo", options=UNIVERSO, format_func=rot_ativo, required=True,
                                                          width="large"),
                "Peso (%)": st.column_config.NumberColumn("Peso (%)", min_value=0.0, max_value=100.0, step=0.5,
                                                          format="%.1f", required=True)})
    pesos_ed = {}
    for t_, v_ in zip(ed["Ativo"], ed["Peso (%)"]):
        if isinstance(t_, str) and t_ in UNIVERSO and pd.notna(v_) and v_ > 0:
            pesos_ed[t_] = pesos_ed.get(t_, 0.0) + float(v_)
    pesos_ed = dict(list(pesos_ed.items())[:USER_PORTFOLIO_MAX])
    with col_bt:
        if st.button("⚖️ Pesos iguais", width="stretch", disabled=not pesos_ed):
            st.session_state["cart_df"] = pd.DataFrame({"Ativo": list(pesos_ed),
                                                        "Peso (%)": [round(100 / len(pesos_ed), 1)] * len(pesos_ed)})
            st.session_state["cart_v"] += 1
            st.rerun()
        if st.button("🔗 Guardar no link", width="stretch", disabled=not pesos_ed,
                     help="Põe a carteira no endereço da página: guarda-o nos favoritos ou partilha-o."):
            st.query_params["carteira"] = ca.format_carteira(pesos_ed)
            st.toast("Carteira guardada no link da página.")
        st.caption("Linha nova: clica na última linha da tabela. Para apagar, seleciona a linha e carrega em Delete.")
        nova = st.text_input("Outra ação (ticker do Yahoo)", key="cart_nova", placeholder="ex.: ASML, SAP.DE, BRK.B",
                             help="Qualquer ação ou ETF do Yahoo Finance. O setor vem da classificação do Yahoo; sem "
                                  "setor (ex.: um ETF de ouro) liga-se ao mercado.")
        if st.button("➕ Acrescentar", width="stretch", disabled=not nova.strip()):
            ficha = _validar(nova)
            if ficha is None:
                st.error(f"O Yahoo não conhece «{nova.strip()}» (ou não respondeu).")
            elif ficha["ticker"] in pesos_ed:
                st.info(f"{ficha['ticker']} já está na carteira.")
            else:
                if ficha["ticker"] not in base:
                    extras[ficha["ticker"]] = ficha
                linhas = pesos_ed | {ficha["ticker"]: 5.0}
                st.session_state["cart_df"] = pd.DataFrame({"Ativo": list(linhas), "Peso (%)": list(linhas.values())})
                st.session_state["cart_v"] += 1
                st.rerun()

    if not pesos_ed:
        st.info("Acrescenta pelo menos um ativo com peso positivo.")
    else:
        w_u = ca.normalizar(pesos_ed)
        soma = sum(pesos_ed.values())
        if abs(soma - 100) > 0.05:
            st.caption(f"Os pesos somam {fm.num(soma, 1)}%: foram ajustados em proporção para somar 100%.")
        try:
            px_u = precos(tuple(dict.fromkeys(list(w_u.index) + [MARKET])), periodo)
            r_u = ck.returns(px_u)
            tab_u, res_u = ca.risco_carteira(r_u[list(w_u.index)], w_u, r_u[MARKET], CVAR_LEVEL, PANIC_Q)
        except Exception as e:  # noqa: BLE001
            st.error(f"Sem preços para esta carteira: {e}")
            tab_u = None
        if tab_u is not None:
            expo = ca.exposicao_setorial(w_u, SET_DE)
            sc_cart = ca.score_carteira(expo, sc["score_final"]) if sc is not None else float("nan")
            u1, u2, u3, u4, u5 = st.columns(5)
            metrica(u1, "Volatilidade anual", fm.pct(res_u["vol"]),
                    f"diversificação {fm.mult(res_u['racio_diversificacao_vol'])}",
                    ajuda="Diversificação = soma dos riscos isolados ÷ risco da carteira. 1× = nenhuma.")
            metrica(u2, f"CVaR {N_CVAR} · 1 dia", fm.pct(res_u["cvar"], 2),
                    f"diversificação {fm.mult(res_u['racio_diversificacao_cvar'])}",
                    ajuda=f"Perda média nos 2,5% piores dias ({janela.lower()}, {res_u['dias']} dias). Histórico.")
            metrica(u3, f"Beta ao {MARKET}", fm.num(res_u["beta"]), f"em pânico {fm.num(res_u['downside_beta'])}",
                    ajuda="Quanto a carteira se move por cada 1% do mercado; 'em pânico' só nos piores dias do SPY.")
            metrica(u4, "Score de risco setorial", fm.num(sc_cart, 0) if sc_cart == sc_cart else "—",
                    "média dos setores, pela exposição",
                    ajuda="Média dos scores (0–100) dos setores em que a carteira está, ponderada pela exposição.")
            top_u = tab_u["pct_cvar"].idxmax()
            metrica(u5, "Quem mais pesa no risco", top_u,
                    f"{fm.pct(tab_u.loc[top_u, 'pct_cvar'], 0)} do CVaR · peso {fm.pct(tab_u.loc[top_u, 'peso'], 0)}")

            # --- previsão oficial da carteira e o seu backtest (FHS sobre os retornos com os pesos atuais)
            bt_u = None
            try:
                px_l = precos(tuple(dict.fromkeys(list(w_u.index))), f"{USER_FORECAST_YEARS}y")
                bt_u = backtest_carteira(ca.format_carteira(pesos_ed), px_l.pct_change().iloc[1:], w_u,
                                         str(px_l.index[-1].date()))
            except Exception:  # noqa: BLE001  sem preços longos ou GARCH sem convergir: fica só o histórico
                pass
            v1, v2, v3, v4 = st.columns(4)
            if bt_u:
                pv = bt_u["previsao"]
                t975 = bt_u["testes"][np.isclose(bt_u["testes"]["nivel"], CVAR_LEVEL)].iloc[0]
                metrica(v1, f"ES {N_CVAR} previsto · {pv['data_previsao']}", fm.pct(pv["es975_aj"], 2),
                        f"modelo {fm.pct(pv['es975'], 2)} × {fm.num(pv['mult_es'])}" if pv["mult_es"] > 1
                        else f"histórico: {fm.pct(res_u['cvar'], 2)}",
                        ajuda=f"Modelo oficial ({OFICIAL}) aplicado aos retornos da carteira com os pesos atuais. Se o "
                              "backtest desta carteira mostrar ES subestimado, o valor já vem corrigido.")
                metrica(v2, "VaR 99% previsto", fm.pct(pv["var99"], 2), f"VaR {N_CVAR}: {fm.pct(pv['var975'], 2)}")
                metrica(v3, "Backtest desta carteira", modelo_txt(pv),
                        f"{int(t975['excecoes'])} exceções em {bt_u['dias']} dias (esperadas {fm.num(t975['esperadas'], 0)})",
                        ajuda=f"O {OFICIAL} previu o VaR e o ES desta carteira, dia a dia, só com dados até à véspera; "
                              f"✅ se passa Kupiec, Christoffersen e Z2 a {N_CVAR}.")
                metrica(v4, f"Semáforo ({BT_TL_DAYS} dias)", SEM[pv["semaforo"]], "exceções ao VaR 99%")
                with st.expander(f"Backtest desta carteira: {bt_u['dias']} dias de previsões"):
                    sb = bt_u["serie"]
                    exc_u = -sb["ret"] > sb["var"]
                    fig_b = go.Figure()
                    fig_b.add_trace(go.Scatter(x=sb["data"], y=sb["ret"], mode="lines", name="Retorno diário da carteira",
                                               line=dict(width=0.8, color=MUTED)))
                    fig_b.add_trace(go.Scatter(x=sb["data"], y=-sb["var"], mode="lines", name=f"−VaR {N_CVAR} previsto",
                                               line=dict(width=1.6, color="#38bdf8")))
                    fig_b.add_trace(go.Scatter(x=sb["data"], y=-sb["es"], mode="lines", name=f"−ES {N_CVAR} previsto",
                                               line=dict(width=1, color="#a855f7", dash="dot")))
                    fig_b.add_trace(go.Scatter(x=sb["data"][exc_u], y=sb["ret"][exc_u], mode="markers", name="Exceção",
                                               marker=dict(size=5, color=RISCO["vermelho"])))
                    layout(fig_b, 340, hovermode="x unified", legend=dict(orientation="h", y=-0.15),
                           yaxis=dict(tickformat=".0%", gridcolor=GRID, zeroline=False), xaxis=dict(gridcolor=GRID))
                    st.plotly_chart(fig_b, width="stretch", config={"displayModeBar": False})
                    tt = bt_u["testes"]

                    def _v(p_: float) -> str:
                        return "❌ rejeita" if p_ < BT_SIGNIF else "✅ passa"

                    st.dataframe(pd.DataFrame({
                        "Nível": tt["nivel"].map(fm.nivel),
                        "Exceções": tt["excecoes"].astype(str) + " (esperadas " + tt["esperadas"].round(0).astype(int).astype(str) + ")",
                        "Kupiec": tt["p_kupiec"].map(_v), "Christoffersen": tt["p_christoffersen"].map(_v),
                        "Z2 (ES)": tt["p_z2"].map(_v), "Z2": tt["z2"].map(lambda v: fm.num(v, 2, sinal=True)),
                        f"Semáforo {BT_TL_DAYS}d": tt["semaforo"].map(SEM),
                    }), hide_index=True, width="stretch")
                    st.caption(f"Os pesos de hoje são aplicados ao passado ({bt_u['dias_precos']} dias em que todas as "
                               f"posições têm preço); as previsões cobrem os últimos {bt_u['dias']} dias. É uma amostra "
                               "curta: com cerca de 2 anos, os testes só apanham falhas grandes, e não inclui nenhuma "
                               "crise como 2008 ou 2020.")
            else:
                v1.caption("Sem previsão nem backtest para a carteira: alguma posição tem menos de 3 anos e meio de "
                           "preços. Ficam as medidas históricas.")

            # --- mapa: setores (tamanho = exposição) com as posições da carteira à volta
            st.subheader("A carteira no mapa de risco")
            posk = {n: np.array(p_, dtype=float) for n, p_ in nx.kamada_kawai_layout(grafo, weight="weight").items()}
            fig_u = go.Figure()
            ux, uy = [], []
            for a, b in grafo.edges():
                ux += [posk[a][0], posk[b][0], None]
                uy += [posk[a][1], posk[b][1], None]
            fig_u.add_trace(go.Scatter(x=ux, y=uy, mode="lines", hoverinfo="skip", showlegend=False,
                                       line=dict(width=2, color="#475569")))
            emax = max(expo[setores].max(), 1e-9)
            fig_u.add_trace(go.Scatter(
                x=[posk[n][0] for n in setores], y=[posk[n][1] for n in setores], mode="markers+text", text=setores,
                textfont=dict(color="#0b1220", size=10), hoverinfo="text", showlegend=False,
                marker=dict(size=[26 + 34 * expo[n] / emax for n in setores],
                            color=[float(sc.loc[n, "score_final"]) if sc is not None else 50.0 for n in setores],
                            colorscale=[[0, RISCO["verde"]], [0.5, RISCO["amarelo"]], [1, RISCO["vermelho"]]],
                            cmin=0, cmax=100, opacity=[1.0 if expo[n] > 0 else 0.3 for n in setores],
                            line=dict(width=1.5, color="#ffffff"),
                            colorbar=dict(title="Score<br>do setor", thickness=10, len=0.5)),
                hovertext=[f"<b>{nome(n)}</b><br>Exposição da carteira: {fm.pct(expo[n])}"
                           + (f"<br>Score {fm.num(sc.loc[n, 'score_final'], 0)}/100" if sc is not None else "")
                           for n in setores]))
            centro = np.mean(list(posk.values()), axis=0)
            sx, sy, stx, shv, ssz, scol, lx, ly = [], [], [], [], [], [], [], []
            for s_ in setores:
                pos_do_setor = [t for t in w_u.index if SET_DE.get(t) == s_ and t != s_]
                if not pos_do_setor:
                    continue
                fora = posk[s_] - centro
                ang0 = np.arctan2(fora[1], fora[0])
                for j, t in enumerate(pos_do_setor):
                    ang = ang0 + (j - (len(pos_do_setor) - 1) / 2) * 0.7
                    q = posk[s_] + 0.42 * np.array([np.cos(ang), np.sin(ang)])
                    sx.append(q[0]); sy.append(q[1]); stx.append(t)   # noqa: E702
                    lx += [posk[s_][0], q[0], None]
                    ly += [posk[s_][1], q[1], None]
                    ssz.append(14 + 20 * w_u[t] / w_u.max())
                    scol.append(float(np.clip(tab_u.loc[t, "racio"], 0.5, 1.5)))
                    shv.append(f"<b>{rot_ativo(t)}</b><br>Peso {fm.pct(w_u[t])} · {fm.pct(tab_u.loc[t, 'pct_cvar'])} do CVaR"
                               f"<br>Pesa {fm.mult(tab_u.loc[t, 'racio'])} no risco face ao capital")
            fig_u.add_trace(go.Scatter(x=lx, y=ly, mode="lines", hoverinfo="skip", showlegend=False,
                                       line=dict(width=1.2, color="#64748b")))
            fig_u.add_trace(go.Scatter(x=sx, y=sy, mode="markers+text", text=stx, textposition="top center",
                                       textfont=dict(color=TXT, size=10), hoverinfo="text", hovertext=shv, showlegend=False,
                                       marker=dict(size=ssz, color=scol, cmin=0.5, cmax=1.5, symbol="diamond",
                                                   colorscale=[[0, RISCO["verde"]], [0.5, RISCO["amarelo"]],
                                                               [1, RISCO["vermelho"]]],
                                                   line=dict(width=1, color="#ffffff"))))
            layout(fig_u, 560, hovermode="closest", dragmode=False, xaxis=dict(visible=False),
                   yaxis=dict(visible=False, scaleanchor="x"))
            st.plotly_chart(fig_u, width="stretch", config={"displayModeBar": False})
            st.caption("**Círculos:** setores; o tamanho é a exposição da carteira e a cor o score de risco do setor (os "
                       "apagados não estão na carteira). **Losangos:** as tuas empresas, junto ao seu setor; o tamanho é o "
                       "peso e a cor diz se pesam mais no risco do que no capital (vermelho) ou menos (verde). As linhas "
                       "entre setores são a espinha de contágio: setores vizinhos caem juntos.")

            c_pos, c_set = st.columns([1, 1])
            with c_pos:
                st.markdown("**Posições**")
                o_u = tab_u.sort_values("pct_cvar", ascending=False)
                st.dataframe(pd.DataFrame({
                    "Ativo": [f"{t} ({SET_DE.get(t, '—')})" if SET_DE.get(t) not in (t, None) else t for t in o_u.index],
                    "Peso": o_u["peso"].map(fm.pct).values,
                    "% do CVaR": o_u["pct_cvar"].map(fm.pct).values,
                    "Risco / capital": o_u["racio"].map(fm.mult).values,
                    f"CVaR {N_CVAR} isolado": o_u["cvar_individual"].map(lambda v: fm.pct(v, 2)).values,
                    "ES previsto": [fm.pct(es_emp(t)) if es_emp(t) is not None
                                    else fm.pct(off.loc[t, "es975"]) if off is not None and t in off.index else "—"
                                    for t in o_u.index],
                    "Beta": o_u["beta"].map(fm.num).values,
                }), hide_index=True, width="stretch")
                st.caption("% do CVaR: parte da perda dos piores dias da carteira que vem desta posição (contribuição de "
                           "Euler; as partes somam 100%). Risco / capital acima de 1× = pesa mais no risco do que no "
                           "dinheiro investido.")
            with c_set:
                st.markdown("**Exposição por setor**")
                e_u = expo[expo > 0].sort_values(ascending=False)
                linhas = []
                for s_, v_ in e_u.items():
                    if s_ in SECTORS and sc is not None:
                        r_ = sc.loc[s_]
                        linhas.append([nome(s_), fm.pct(v_), int(round(r_["score_final"])),
                                       pi_level(r_["probabilidade"], r_["impacto"]),
                                       SEM[r_["semaforo"]].split()[0] + (" ⚠" if r_["risco_modelo"] else "")])
                    else:
                        linhas.append([s_ if s_ not in SECTORS else nome(s_), fm.pct(v_), None, "—", "—"])
                st.dataframe(pd.DataFrame(linhas, columns=["Setor", "Exposição", "Score", "Nível", "Modelo"]),
                             hide_index=True, width="stretch", column_config={
                                 "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d")})
                crit_u = [s_ for s_ in e_u.index if s_ in SECTORS and sc is not None
                          and sc.loc[s_, "impacto"] * sc.loc[s_, "probabilidade"] >= CRIT_MIN]
                if crit_u:
                    st.warning(f"{fm.pct(e_u[crit_u].sum(), 0)} da carteira está em setores de nível {CRIT}: "
                               f"{', '.join(crit_u)}.")
                al_u = [a for a in (alertas or {}).get("alertas", [])
                        if any(f"[[{s_}]]" in a["texto"] for s_ in e_u.index)]
                for a in al_u:
                    st.markdown(f"🔔 {a['texto'].replace('[[', '').replace(']]', '')}")

            if len(w_u) >= 2:
                with st.expander("🛡️ Sugestão defensiva: carteira HRP (dependência de cauda)"):
                    w_h = ca.calcular_pesos_hrp(r_u[list(w_u.index)])
                    if w_h is None:
                        st.caption(f"Sem dados para calcular: são precisos {HRP_MIN_DAYS} dias em que todas as posições "
                                   "tenham preço. Escolhe uma janela maior ou tira a posição mais recente.")
                    else:
                        _, res_h = ca.risco_carteira(r_u[list(w_u.index)], w_h, r_u[MARKET], CVAR_LEVEL, PANIC_Q)
                        h1, h2, h3 = st.columns(3)
                        metrica(h1, f"CVaR {N_CVAR} · 1 dia", fm.pct(res_h["cvar"], 2),
                                f"atual {fm.pct(res_u['cvar'], 2)} ({fm.pct(res_h['cvar'] / res_u['cvar'] - 1, 0, sinal=True)})")
                        metrica(h2, "Volatilidade anual", fm.pct(res_h["vol"]),
                                f"atual {fm.pct(res_u['vol'])} ({fm.pct(res_h['vol'] / res_u['vol'] - 1, 0, sinal=True)})")
                        metrica(h3, "Diversificação", fm.mult(res_h["racio_diversificacao_vol"]),
                                f"atual {fm.mult(res_u['racio_diversificacao_vol'])}")
                        st.dataframe(pd.DataFrame({
                            "Ativo": [rot_ativo(t) for t in w_u.index],
                            "Peso atual": w_u.map(lambda v: fm.pct(v, 1)).values,
                            "Peso HRP": w_h.map(lambda v: fm.pct(v, 1)).values,
                            "Diferença": (w_h - w_u).map(lambda v: fm.pp(100 * v, 1)).values,
                        }), hide_index=True, width="stretch")
                        if st.button("Usar estes pesos na tabela"):
                            st.session_state["cart_df"] = pd.DataFrame({"Ativo": list(w_h.index),
                                                                        "Peso (%)": [round(100 * v, 1) for v in w_h]})
                            st.session_state["cart_v"] += 1
                            st.rerun()
                        st.caption("Hierarchical Risk Parity (López de Prado, 2016): junta os ativos que caem nos mesmos dias "
                                   "(os 5% piores de cada um), ordena-os por essa proximidade e reparte o peso ao meio, vezes "
                                   f"sem conta, dando mais ao lado com menos CVaR {N_CVAR}. Usa só os ativos que já tens e a "
                                   f"mesma janela ({janela.lower()}). É um exercício com dados passados, calculado na mesma "
                                   "amostra em que é avaliado, por isso a redução mostrada é otimista. Não considera retornos "
                                   "esperados, custos nem impostos, e não é aconselhamento de investimento.")

            st.subheader("Stress da carteira")
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Cenários históricos (aproximação pelo setor)**")
                if stt is not None:
                    ss_u = ca.stress_setorial(expo, stt)
                    st.dataframe(pd.DataFrame({
                        "Cenário": ss_u["cenario"],
                        "Carteira": (-ss_u["perda"]).map(fm.pct),
                        MARKET: (-ss_u["perda_mercado"]).map(fm.pct),
                        "Setor que mais pesa": ss_u["maior_contribuinte"],
                    }), hide_index=True, width="stretch")
                    st.caption("Cada posição leva a perda que o seu setor teve no episódio. Ignora o risco próprio de cada "
                               "empresa, por isso subestima carteiras concentradas em poucas ações.")
            with g2:
                st.markdown(f"**Choque no mercado ({MARKET})**")
                lo, hi, dflt, stp = SHOCK_MARKET
                ch_u = st.slider(f"Variação do {MARKET}", lo, hi, dflt, stp, format="%.0f%%", key="cart_choque")
                r_ch_u = float((w_u * shock_returns(tab_u["beta"], ch_u / 100)).sum())
                metrica(st, f"Carteira com o {MARKET} a {fm.num(ch_u, 0, sinal=True)}%", fm.pct(r_ch_u, 1),
                        f"{fm.mult(abs(r_ch_u) / res_u['cvar'], 1)} o CVaR diário da carteira")
                st.caption("Retorno = beta de cada posição × choque (betas da janela escolhida). Aproximação linear, sem "
                           "o contágio próprio das crises.")
            st.caption(f"As medidas de risco e as contribuições são históricas ({janela.lower()} de preços). O ES e o VaR "
                       f"previstos vêm do modelo oficial ({OFICIAL}). Não é aconselhamento de investimento.")
