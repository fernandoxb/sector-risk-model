"""Análise ao vivo: CVaR histórico, downside beta, regime de juros e exportação para o Obsidian.

Os dados do run chegam em `g` (as variáveis do app.py).
"""
import os
import pandas as pd
import streamlit as st
from datetime import date
from pathlib import Path
from config.settings import (CVAR_LEVEL, LINK_T_MIN, OBSIDIAN_FOLDER, OBSIDIAN_VAULT, PANIC_Q, RATE_LOOKBACK,
                             RATE_THRESHOLD_PP, SECTORS, SECTOR_PROFILE, SHOCK_DIFF)
from src.models import cockpit as ck
from src.report import formato as fm
from paineis.comum import N_CVAR, metrica


def render(g: dict) -> None:
    links = g["links"]
    grafo = g["grafo"]
    janela = g["janela"]
    mkt = g["mkt"]
    regime = g["regime"]
    rets = g["rets"]

    st.caption(f"Calculado agora, com os preços de {rets.index[0].date()} a {rets.index[-1].date()} "
               f"(janela de {janela.lower()} escolhida na barra lateral).")
    # β medido de cada setor face aos juros (regressão semanal HAC): só as ligações significativas contam
    if links is not None:
        lj = links[links["fator"] == "juros"].set_index("setor")
        b_juros = lj["beta"].where(lj["aceite"].astype(bool))
        t_juros = lj["t"]
    else:
        b_juros, t_juros = pd.Series(dtype=float), pd.Series(dtype=float)
    sofrem = sorted(b_juros[b_juros < 0].index, key=lambda t: b_juros[t])
    with st.container(border=True):
        p1, p2, p3 = st.columns(3)
        alpha = p1.slider("Nível do CVaR", 0.900, 0.995, CVAR_LEVEL, step=0.005, format="%.3f",
                          help=f"Padrão do projeto: {N_CVAR} (nível do ES no FRTB).")
        panic_q = p2.slider("Dias de pânico do SPY", 0.05, 0.30, PANIC_Q, step=0.05,
                            help="Fração dos piores dias do SPY usada no downside beta (β↓).")
        choque = p3.slider("Choque de juros (p.p.)", SHOCK_DIFF[0], SHOCK_DIFF[1], 0.0, SHOCK_DIFF[3],
                           format="%+.2f p.p.", disabled=b_juros.dropna().empty,
                           help="Simula uma subida (ou descida) do Treasury 10Y. A perda de cada setor vem do seu β "
                                f"medido; só contam os setores com ligação significativa (|t| ≥ {fm.num(LINK_T_MIN, 0)}): "
                                f"{', '.join(sofrem) or 'nenhum'} sofrem quando os juros sobem.")
    nivel_live = fm.nivel(alpha)

    tab = ck.tail_table(rets, mkt, alpha, panic_q)
    tab["cvar_aj"] = ck.rate_stress(tab["cvar"], b_juros, choque)
    n_panico = int((mkt <= mkt.quantile(panic_q)).sum())

    c1, c2, c3, c4 = st.columns(4)
    top = tab["cvar_aj"].idxmax()
    metrica(c1, f"Maior CVaR histórico ({nivel_live})", top,
            fm.pct(tab.loc[top, "cvar_aj"], 2) + (" com o choque de juros" if choque else ""),
            ajuda="Perda média nos piores dias da janela; com um choque de juros, soma a perda estimada pelo β medido.")
    amp = tab["downside_beta"].idxmax()
    metrica(c2, "Maior amplificador em pânico", amp, f"β↓ = {fm.num(tab.loc[amp, 'downside_beta'])}",
            ajuda=f"Downside beta: quanto o setor se move por cada 1% do SPY nos {n_panico} piores dias do SPY. "
                  "Acima de 1 amplifica as quedas do mercado.")
    if regime:
        metrica(c3, f"Treasury 10Y ({regime['asof']})", fm.pct(regime["nivel"] / 100, 2),
                f"{fm.pp(regime['delta'])} em {RATE_LOOKBACK} dias úteis")
        metrica(c4, "Regime de juros", regime["regime"].capitalize(),
                f"choque simulado {fm.pp(choque)}" if choque else None,
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
        "CVaR com choque de juros": tab["cvar_aj"].map(lambda v: fm.pct(v, 2)).values,
        "Downside beta": tab["downside_beta"].map(fm.num).values,
        "Beta": tab["beta"].map(fm.num).values,
        "Beta aos juros (%/p.p.)": [fm.num(100 * lj.loc[t, "beta"], 1, sinal=True) + ("" if pd.notna(b_juros.get(t)) else " (n.s.)")
                                    if links is not None and t in lj.index else "—" for t in tab.index],
        "t (HAC)": [fm.num(t_juros[t], 1, sinal=True) if t in t_juros.index else "—" for t in tab.index],
    })
    with st.expander(f"📋 Tabela: CVaR {nivel_live}, betas e sensibilidade aos juros"):
        st.dataframe(tabela.iloc[tab["cvar_aj"].values.argsort()[::-1]], hide_index=True, width="stretch")
        st.caption("Beta aos juros: retorno semanal do setor (%) por +1 p.p. no Treasury 10Y, medido por regressão com "
                   f"erros HAC nos últimos 3 anos; (n.s.) = não significativo (|t| < {fm.num(LINK_T_MIN, 0)}), não entra "
                   "no choque. CVaR com choque = CVaR + perda estimada do choque, só quando é perda. Tipo: classificação "
                   "económica do setor, não um nível de risco.")

    st.subheader("🗂️ Exportar para o Obsidian")
    # a pasta vem só da configuração: uma caixa de texto deixaria um visitante do site escolher onde gravar
    vault = Path(os.environ.get("OBSIDIAN_VAULT", str(OBSIDIAN_VAULT)))
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
