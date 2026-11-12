"""Análise ao vivo: CVaR histórico, downside beta, regime de juros e exportação para o Obsidian.

Gerado a partir do antigo app.py: o código do separador é o mesmo; os dados do run chegam em `g`.
"""
import os
import pandas as pd
import streamlit as st
from datetime import date
from pathlib import Path
from config.settings import (CVAR_LEVEL, GROWTH_SECTORS, OBSIDIAN_FOLDER, OBSIDIAN_VAULT, PANIC_Q, RATE_LOOKBACK,
                             RATE_THRESHOLD_PP, SECTORS, SECTOR_PROFILE)
from src.models import cockpit as ck
from src.report import formato as fm
from paineis.comum import N_CVAR, metrica


def render(g: dict) -> None:
    d_yield = g["d_yield"]
    grafo = g["grafo"]
    janela = g["janela"]
    mkt = g["mkt"]
    regime = g["regime"]
    rets = g["rets"]

    st.caption(f"Calculado agora, com os preços de {rets.index[0].date()} a {rets.index[-1].date()} "
               f"(janela de {janela.lower()} escolhida na barra lateral).")
    with st.container(border=True):
        p1, p2, p3, p4 = st.columns(4)
        alpha = p1.slider("Nível do CVaR", 0.900, 0.995, CVAR_LEVEL, step=0.005, format="%.3f",
                          help=f"Padrão do projeto: {N_CVAR} (nível do ES no FRTB).")
        panic_q = p2.slider("Dias de pânico do SPY", 0.05, 0.30, PANIC_Q, step=0.05,
                            help="Fração dos piores dias do SPY usada no downside beta (β↓).")
        usar_juros = p3.toggle("Ajuste de juros", value=True,
                               help=f"Penaliza os setores de crescimento ({', '.join(GROWTH_SECTORS)}) quando os "
                                    "juros sobem: CVaR ajustado = CVaR × "
                                    "(1 + penalização × subida dos juros em p.p.).")
        penal = p3.slider("Penalização por +1 p.p.", 0.0, 2.0, 0.5, step=0.1, disabled=not usar_juros)
        choque = p4.slider("Choque adicional de juros (p.p.)", -1.0, 2.0, 0.0, step=0.25, disabled=not usar_juros,
                           help="Soma-se à variação real dos últimos ~3 meses, para simular um cenário.")
    nivel_live = fm.nivel(alpha)

    tab = ck.tail_table(rets, mkt, alpha, panic_q, d_yield)
    delta_cen = (regime["delta"] if regime else 0.0) + choque
    tab["cvar_aj"] = (ck.rate_adjusted(tab["cvar"], delta_cen, GROWTH_SECTORS, penal)
                      if usar_juros and regime else tab["cvar"])
    n_panico = int((mkt <= mkt.quantile(panic_q)).sum())

    c1, c2, c3, c4 = st.columns(4)
    top = tab["cvar_aj"].idxmax()
    metrica(c1, f"Maior CVaR histórico ({nivel_live})", top, fm.pct(tab.loc[top, "cvar_aj"], 2),
            ajuda="Perda média nos piores dias da janela, já com o ajuste de juros se estiver ativo.")
    amp = tab["downside_beta"].idxmax()
    metrica(c2, "Maior amplificador em pânico", amp, f"β↓ = {fm.num(tab.loc[amp, 'downside_beta'])}",
            ajuda=f"Downside beta: quanto o setor se move por cada 1% do SPY nos {n_panico} piores dias do SPY. "
                  "Acima de 1 amplifica as quedas do mercado.")
    if regime:
        metrica(c3, f"Treasury 10Y ({regime['asof']})", fm.pct(regime["nivel"] / 100, 2),
                f"{fm.pp(regime['delta'])} em {RATE_LOOKBACK} dias úteis")
        metrica(c4, "Regime de juros", regime["regime"].capitalize(),
                f"cenário {fm.pp(delta_cen)}" if choque else None,
                ajuda=f"Subida se os juros subiram ≥ {fm.num(RATE_THRESHOLD_PP)} p.p. em {RATE_LOOKBACK} dias úteis, "
                      f"descida se caíram ≥ {fm.num(RATE_THRESHOLD_PP)} p.p., estável no meio.")
    else:
        c3.caption("FRED indisponível: regime de juros desligado.")

    st.caption("A rede de contágio entre setores e as empresas de cada setor estão no separador **Mapa de risco**.")

    tabela = pd.DataFrame({
        "Ticker": tab.index,
        "Setor": [SECTORS[t] for t in tab.index],
        "Tipo": [SECTOR_PROFILE[t]["tipo"] for t in tab.index],
        "CVaR": tab["cvar"].map(lambda v: fm.pct(v, 2)).values,
        "CVaR ajustado": tab["cvar_aj"].map(lambda v: fm.pct(v, 2)).values,
        "Downside beta": tab["downside_beta"].map(fm.num).values,
        "Beta": tab["beta"].map(fm.num).values,
        "Beta aos juros (%/p.p.)": (tab["beta_juros"].map(lambda v: fm.num(v * 100, 1, sinal=True))
                                    if "beta_juros" in tab else ["—"] * len(tab)),
    })
    with st.expander(f"📋 Tabela: CVaR {nivel_live}, betas e sensibilidade aos juros"):
        st.dataframe(tabela.iloc[tab["cvar_aj"].values.argsort()[::-1]], hide_index=True, width="stretch")
        st.caption("Beta aos juros: variação diária média do setor (%) por +1 p.p. no Treasury 10Y. Tipo: classificação "
                   "económica do setor (cíclico, sensível, defensivo), não um nível de risco.")

    st.subheader("🗂️ Exportar para o Obsidian")
    vault = Path(st.text_input("Pasta do cofre", os.environ.get("OBSIDIAN_VAULT", str(OBSIDIAN_VAULT))))
    hoje = date.today()
    params = {"janela": janela, "confianca_cvar": alpha, "limiar_panico": panic_q,
              "dados_ate": str(rets.index[-1].date())}
    edges = [(a, b, d["corr"]) for a, b, d in sorted(grafo.edges(data=True), key=lambda e: e[2]["weight"])]
    nota = ck.obsidian_log(hoje, params, tabela.drop(columns="Tipo"), edges, regime)
    destino = vault / OBSIDIAN_FOLDER / f"{hoje.isoformat()}.md"
    c_a, c_b = st.columns(2)
    if not vault.is_dir():                       # ex.: Streamlit Cloud, sem cofre local
        c_a.caption("Cofre não encontrado neste computador: usa o botão para descarregar a nota.")
    elif c_a.button("💾 Gravar a nota de hoje no cofre", type="primary"):
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(nota, encoding="utf-8")
        st.success(f"Gravado em {destino}")
    c_b.download_button("⬇️ Descarregar .md", nota, file_name=destino.name, mime="text/markdown")
    with st.expander("Pré-visualizar a nota"):
        st.code(nota, language="markdown")
