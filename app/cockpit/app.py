"""Cockpit de Risco Setorial Interativo.

Correr a partir da raiz do projeto:
    .venv/bin/streamlit run app/cockpit/app.py
Metodologia: docs/nota-metodologica.md
"""
import os
import sys
from datetime import date
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.settings import (CVAR_LEVEL, CVAR_YEARS, GROWTH_SECTORS, MARKET,  # noqa: E402
                             OBSIDIAN_FOLDER, OBSIDIAN_VAULT, PANIC_Q, RATE_LOOKBACK,
                             RATE_SERIES, SECTOR_PROFILE, SECTORS)
from src.models import cockpit as ck  # noqa: E402

st.set_page_config(page_title="Cockpit de Risco Setorial", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

CORES = {"Cíclico": "#ef4444", "Sensível": "#f97316", "Defensivo": "#10b981"}
BG = "#0e1117"

# ============================================================================ controlos
st.sidebar.title("🎮 Painel de Controlo")
st.sidebar.caption("Ajusta os parâmetros para testar o stress do mercado.")
janela = st.sidebar.selectbox("📅 Janela histórica", ["1 Ano", "2 Anos", "3 Anos"],
                              index=min(max(CVAR_YEARS, 1), 3) - 1)
periodo = {"1 Ano": "1y", "2 Anos": "2y", "3 Anos": "3y"}[janela]
alpha = st.sidebar.slider("🎯 Confiança do CVaR", 0.900, 0.995, CVAR_LEVEL, step=0.005, format="%.3f",
                          help="Padrão do projeto (config/settings.py): igual ao mapa de riscos.")
panic_q = st.sidebar.slider("🚨 Limiar de pânico do SPY (percentil)", 0.05, 0.30, PANIC_Q, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🏦 Regime de juros (10Y)")
usar_juros = st.sidebar.toggle("Penalizar crescimento em subida de juros", value=True)
penal = st.sidebar.slider("Penalização do CVaR por +1 p.p.", 0.0, 2.0, 0.5, step=0.1,
                          disabled=not usar_juros)
choque = st.sidebar.slider("Choque adicional de juros (p.p.)", -1.0, 2.0, 0.0, step=0.25,
                           disabled=not usar_juros,
                           help="Soma-se à variação real dos últimos ~3 meses para simular cenários.")


# ============================================================================ dados (cache)
@st.cache_data(ttl=3600, show_spinner="🤖 A descarregar preços dos setores…")
def precos(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return ck.download_close(list(tickers), period)


@st.cache_data(ttl=3600, show_spinner="🏦 A ler o Treasury 10Y no FRED…")
def juros(start: str) -> pd.Series:
    return ck.fetch_rates(RATE_SERIES, start)


setores = list(SECTORS)
try:
    rets_all = ck.returns(precos(tuple(setores + [MARKET]), periodo))
except Exception as e:  # noqa: BLE001
    st.error(f"Não foi possível descarregar os preços do Yahoo: {e}")
    st.stop()
rets, mkt = rets_all[setores], rets_all[MARKET]

regime, d_yield = None, None
try:
    y = juros((rets.index[0] - pd.Timedelta(days=200)).date().isoformat())
    regime = ck.rate_regime(y, RATE_LOOKBACK)
    d_yield = y.diff().reindex(rets.index)
except Exception as e:  # noqa: BLE001
    st.sidebar.warning(f"FRED indisponível ({e}); regime de juros desligado.")

tab = ck.tail_table(rets, mkt, alpha, panic_q, d_yield)
delta_cen = (regime["delta"] if regime else 0.0) + choque
tab["cvar_aj"] = (ck.rate_adjusted(tab["cvar"], delta_cen, GROWTH_SECTORS, penal)
                  if usar_juros and regime else tab["cvar"])
grafo, corr = ck.mst(rets)

# ============================================================================ cabeçalho
st.title("🛡️ Cockpit de Risco Setorial")
st.caption(f"Dados de {rets.index[0].date()} a {rets.index[-1].date()} · CVaR {alpha:.0%} · "
           f"pânico = {panic_q:.0%} piores dias do SPY ({int((mkt <= mkt.quantile(panic_q)).sum())} dias)")

c1, c2, c3, c4 = st.columns(4)
top = tab["cvar_aj"].idxmax()
c1.metric("Maior risco de cauda", f"{top}", f"CVaR {tab.loc[top, 'cvar_aj']:.2%}", delta_color="off")
amp = tab["downside_beta"].idxmax()
c2.metric("Maior amplificador em pânico", amp, f"β↓ {tab.loc[amp, 'downside_beta']:.2f}", delta_color="off")
if regime:
    c3.metric(f"Treasury 10Y ({regime['asof']})", f"{regime['nivel']:.2f}%",
              f"{regime['delta']:+.2f} p.p. em 3 meses", delta_color="inverse")
    c4.metric("Regime de juros", regime["regime"].capitalize(),
              f"cenário {delta_cen:+.2f} p.p." if choque else None, delta_color="off")

# ============================================================================ grafo MST
st.subheader("🕸️ Espinha de contágio (Minimum Spanning Tree)")
pos = nx.kamada_kawai_layout(grafo, weight="weight")

edge_x, edge_y, mid_x, mid_y, mid_txt = [], [], [], [], []
for a, b, d in grafo.edges(data=True):
    (x0, y0), (x1, y1) = pos[a], pos[b]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
    mid_x.append((x0 + x1) / 2)
    mid_y.append((y0 + y1) / 2)
    mid_txt.append(f"{a} ↔ {b}<br>r = {d['corr']:.2f}<br>d = {d['weight']:.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none",
                         line=dict(width=2, color="#4b5563"), showlegend=False))
fig.add_trace(go.Scatter(x=mid_x, y=mid_y, mode="markers", hoverinfo="text", hovertext=mid_txt,
                         marker=dict(size=10, color="rgba(0,0,0,0)"), showlegend=False))
cmax = tab["cvar_aj"].max()
for tipo, cor in CORES.items():
    nos = [n for n in grafo.nodes if SECTOR_PROFILE[n]["tipo"] == tipo]
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nos], y=[pos[n][1] for n in nos], mode="markers+text", name=tipo,
        text=nos, textposition="bottom center", textfont=dict(color="#e5e7eb"),
        marker=dict(size=[20 + 50 * tab.loc[n, "cvar_aj"] / cmax for n in nos], color=cor,
                    opacity=1, line=dict(width=2, color="#ffffff")),
        hoverinfo="text",
        hovertext=[f"<b>{SECTORS[n]} ({n})</b><br>CVaR: {tab.loc[n, 'cvar']:.2%}"
                   + (f"<br>CVaR ajustado juros: {tab.loc[n, 'cvar_aj']:.2%}"
                      if tab.loc[n, "cvar_aj"] != tab.loc[n, "cvar"] else "")
                   + f"<br>Downside Beta: {tab.loc[n, 'downside_beta']:.2f}"
                   + f"<br>Ligações MST: {grafo.degree(n)}" for n in nos]))
fig.update_layout(height=520, margin=dict(b=0, l=0, r=0, t=10), paper_bgcolor=BG, plot_bgcolor=BG,
                  hovermode="closest",
                  legend=dict(orientation="h", y=-0.02, yanchor="top", font=dict(color="#e5e7eb")),
                  xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
st.plotly_chart(fig, use_container_width=True)
st.caption("Tamanho do nó = CVaR (ajustado ao regime de juros). Distância d = √(2(1−r)): "
           "setores que caem juntos ficam próximos. Passa o rato sobre as ligações para ver r.")

# ============================================================================ tabela setorial
tabela = pd.DataFrame({
    "Ticker": tab.index,
    "Setor": [SECTORS[t] for t in tab.index],
    "Tipo": [SECTOR_PROFILE[t]["tipo"] for t in tab.index],
    "CVaR": tab["cvar"].map("{:.2%}".format).values,
    "CVaR ajustado": tab["cvar_aj"].map("{:.2%}".format).values,
    "Downside Beta": tab["downside_beta"].map("{:.2f}".format).values,
    "Beta": tab["beta"].map("{:.2f}".format).values,
    "Beta juros (%/p.p.)": (tab["beta_juros"].map(lambda v: f"{v * 100:+.1f}") if "beta_juros" in tab
                            else ["—"] * len(tab)),
})
with st.expander("📋 Tabela de risco setorial", expanded=False):
    st.dataframe(tabela.sort_values("CVaR ajustado", key=lambda s: s.str.rstrip("%").astype(float),
                                    ascending=False), hide_index=True, use_container_width=True)
    st.caption("Beta juros: variação diária média do setor (%) por +1 p.p. no Treasury 10Y. "
               f"Setores de crescimento penalizados: {', '.join(GROWTH_SECTORS)}.")

# ============================================================================ drill-down
st.markdown("---")
st.subheader("🔍 Microscópio de ativos")
sel = st.selectbox("Setor", setores, format_func=lambda t: f"{SECTORS[t]} ({t})")
acoes = SECTOR_PROFILE[sel]["acoes"]

drill = pd.DataFrame()
try:
    r_ac = ck.returns(precos(tuple(acoes + [sel, MARKET]), periodo))
    t_ac = ck.tail_table(r_ac[acoes], r_ac[MARKET], alpha, panic_q)
    pai = ck.cvar(r_ac[sel], alpha)
    drill = pd.DataFrame({
        "Ticker": acoes,
        "CVaR": t_ac["cvar"].map("{:.2%}".format).values,
        "Downside Beta": t_ac["downside_beta"].map("{:.2f}".format).values,
        "CVaR / setor": (t_ac["cvar"] / pai).map("{:.2f}×".format).values,
        "Diagnóstico": ["⚠️ Risco superior ao setor" if v > pai else "🟢 Risco controlado"
                        for v in t_ac["cvar"]],
    })
    col_t, col_h = st.columns([3, 2])
    with col_t:
        st.dataframe(drill, hide_index=True, use_container_width=True)
        st.caption(f"CVaR {alpha:.0%} do ETF {sel} no mesmo período: {pai:.2%}.")
    with col_h:
        cm = r_ac[acoes].corr()
        hm = px.imshow(cm, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                       aspect="auto")
        hm.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor=BG,
                         font=dict(color="#e5e7eb"), title=dict(text="Correlação entre ações", x=0))
        st.plotly_chart(hm, use_container_width=True)
except Exception as e:  # noqa: BLE001
    st.error(f"Erro ao processar as ações de {sel}: {e}")

# ============================================================================ Obsidian
st.markdown("---")
st.subheader("🗂️ Log diário para o Obsidian")
vault = Path(st.text_input("Pasta do cofre", os.environ.get("OBSIDIAN_VAULT", str(OBSIDIAN_VAULT))))
hoje = date.today()
params = {"janela": janela, "confianca_cvar": alpha, "limiar_panico": panic_q,
          "dados_ate": str(rets.index[-1].date())}
edges = [(a, b, d["corr"]) for a, b, d in sorted(grafo.edges(data=True), key=lambda e: e[2]["weight"])]
nota = ck.obsidian_log(hoje, params, tabela.drop(columns="Tipo"), edges, regime, sel, drill)
destino = vault / OBSIDIAN_FOLDER / f"{hoje.isoformat()}.md"
c_a, c_b = st.columns([1, 1])
if c_a.button("💾 Exportar estado de hoje", type="primary"):
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(nota, encoding="utf-8")
    st.success(f"Gravado em {destino}")
c_b.download_button("⬇️ Descarregar .md", nota, file_name=destino.name, mime="text/markdown")
with st.expander("Pré-visualizar nota"):
    st.code(nota, language="markdown")
