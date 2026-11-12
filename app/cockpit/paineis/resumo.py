"""Resumo: alertas do dia, setores de maior risco, previsão oficial e carteira-base.

Gerado a partir do antigo app.py: o código do separador é o mesmo; os dados do run chegam em `g`.
"""
import pandas as pd
import streamlit as st
from config.settings import CVAR_YEARS
from src.report import formato as fm
from src.report.alerts import pi_level
from paineis.comum import N_CVAR, OFICIAL, SEM, SEV, metrica, nome


def render(g: dict) -> None:
    PIPE_ASOF = g["PIPE_ASOF"]
    PREV_DATA = g["PREV_DATA"]
    alertas = g["alertas"]
    off = g["off"]
    rc = g["rc"]
    sc = g["sc"]
    stt = g["stt"]

    st.caption(f"Dados de mercado até **{PIPE_ASOF}** · previsão para **{PREV_DATA}** · modelo oficial "
               f"**{OFICIAL}** (escolhido pelo backtest de 2008–2026).")
    if sc is None or off is None:
        st.info("Ainda não há resultados do pipeline. Corre `python run_pipeline.py --offline`.")
    else:
        st.subheader("O que mudou")
        if alertas is None:
            st.caption("Sem registo de alertas (o bot Sentinela ainda não correu).")
        else:
            ref = alertas.get("referencia")
            st.caption(f"Alertas do bot Sentinela: {alertas['asof']} comparado com {ref or 'nada (primeiro run)'}.")
            if not alertas["alertas"]:
                st.success("Sem alertas: nenhum semáforo, nível de risco, ligação ou regime mudou.")
            for a in alertas["alertas"]:
                cor, rot = SEV[a["severidade"]]
                c1, c2 = st.columns([1, 9], vertical_alignment="center")
                c1.badge(rot, color=cor)
                texto = a["texto"].replace("[[", "").replace("]]", "")
                tipo = a["tipo"][0].upper() + a["tipo"][1:]
                c2.markdown(texto if texto.lower().startswith(a["tipo"].lower()) else f"**{tipo}** · {texto}")

        st.subheader("Setores de maior risco")
        cols = st.columns(3)
        for col, t in zip(cols, sc.index[:3]):
            r = sc.loc[t]
            nv = pi_level(r["probabilidade"], r["impacto"])
            with col.container(border=True):
                st.markdown(f"**{int(r['rank'])}. {nome(t)}**")
                metrica(st, "Score de risco (0–100)", fm.num(r["score_final"], 0),
                        f"P{int(r['probabilidade'])} × I{int(r['impacto'])} = "
                        f"{int(r['probabilidade'] * r['impacto'])} · nível {nv}")
                st.markdown(f"ES {N_CVAR} previsto para {PREV_DATA}: **{fm.pct(off.loc[t, 'es975'])}**  \n"
                            f"Semáforo do modelo: {SEM[r['semaforo']]}"
                            + (" · ⚠ risco de modelo" if r["risco_modelo"] else ""))

        st.subheader(f"Previsão oficial para {PREV_DATA} ({OFICIAL}, perda a 1 dia)")
        prev = off.loc[sc.index].assign(Setor=[nome(t) for t in sc.index])
        st.dataframe(pd.DataFrame({
            "Setor": prev["Setor"],
            "VaR 99%": prev["var99"].map(fm.pct),
            f"VaR {N_CVAR}": prev["var975"].map(fm.pct),
            f"ES {N_CVAR}": prev["es975"].map(fm.pct),
            "Score": sc["score_final"].round(0).astype(int),
        }), hide_index=True, width="stretch", column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d")})
        st.caption(f"**VaR 99%**: a perda de um dia só deve ultrapassar este valor em 1 dia em cada 100. "
                   f"**ES {N_CVAR}**: a perda média nos 2,5% piores dias. Prevê-se com dados até {PIPE_ASOF} "
                   "(volatilidade do GARCH + cauda empírica dos resíduos).")
        with st.expander("Porque é que o ES previsto é diferente do CVaR histórico da Análise ao vivo?"):
            st.markdown(
                f"São duas perguntas diferentes, ambas a {N_CVAR}:\n"
                f"- **ES previsto ({OFICIAL})**: a perda média esperada **amanhã** nos piores dias, dada a "
                "volatilidade **atual**. Sobe logo quando o mercado fica agitado.\n"
                f"- **CVaR histórico**: a média dos piores dias **dos últimos {CVAR_YEARS} anos**, todos com o mesmo "
                "peso. Muda devagar.\n\n"
                "Quando a volatilidade de hoje está acima da média dos últimos anos, o ES previsto fica acima do "
                "CVaR histórico (é o caso do XLK). O backtest mostra que o previsto acerta melhor.")

        if rc is not None and stt is not None:
            st.subheader("Carteira dos 11 setores (pesos iguais)")
            cart = rc.loc["Carteira"]
            pior = stt[stt["serie"] == "Carteira"].sort_values("perda").iloc[-1]
            k1, k2, k3 = st.columns(3)
            metrica(k1, f"Volatilidade anual · {CVAR_YEARS} anos", fm.pct(cart["contrib_vol"]))
            metrica(k2, f"CVaR {N_CVAR} · 1 dia · {CVAR_YEARS} anos", fm.pct(cart["contrib_cvar"], 2),
                    ajuda="Perda média da carteira nos 2,5% piores dias dos últimos 3 anos.")
            metrica(k3, "Pior cenário de stress", fm.pct(-pior["perda"]), pior["cenario"],
                    ajuda="Retorno acumulado da carteira do pico ao vale do SPY nesse episódio.")
