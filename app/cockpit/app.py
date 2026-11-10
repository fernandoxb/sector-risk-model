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
from config.settings import (BT_LEVELS, BT_SIGNIF, BT_TL_DAYS, BT_WINDOW,  # noqa: E402
                             CVAR_LEVEL, CVAR_YEARS, FEATURES, GROWTH_SECTORS, MARKET,
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
st.plotly_chart(fig, width="stretch")
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
                                    ascending=False), hide_index=True, width="stretch")
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
        st.dataframe(drill, hide_index=True, width="stretch")
        st.caption(f"CVaR {alpha:.0%} do ETF {sel} no mesmo período: {pai:.2%}.")
    with col_h:
        cm = r_ac[acoes].corr()
        hm = px.imshow(cm, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                       aspect="auto")
        hm.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor=BG,
                         font=dict(color="#e5e7eb"), title=dict(text="Correlação entre ações", x=0))
        st.plotly_chart(hm, width="stretch")
except Exception as e:  # noqa: BLE001
    st.error(f"Erro ao processar as ações de {sel}: {e}")

# ============================================================================ backtesting do VaR
st.markdown("---")
st.subheader("🧪 Backtesting do VaR e do ES")
METODOS = {"historico": "Histórico", "garch": "GJR-GARCH(1,1)-t", "fhs": "GARCH-FHS"}
SEMAFORO = {"verde": "🟢 verde", "amarelo": "🟡 amarelo", "vermelho": "🔴 vermelho"}
bt_csv, bt_ser = FEATURES / "var_backtest.csv", FEATURES / "var_backtest_series.parquet"


@st.cache_data(show_spinner=False)
def backtest_dados(mtime: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(bt_csv), pd.read_parquet(bt_ser)


if not (bt_csv.exists() and bt_ser.exists()):
    st.info("Ainda não há resultados de backtesting. Corre `python run_pipeline.py --offline`.")
else:
    bt_res, bt_s = backtest_dados(bt_csv.stat().st_mtime)
    b1, b2, b3 = st.columns(3)
    bt_sel = b1.selectbox("Setor", setores, index=setores.index(sel),
                          format_func=lambda t: f"{SECTORS[t]} ({t})", key="bt_setor")
    bt_met = b2.radio("Método", list(METODOS), format_func=METODOS.get, horizontal=True)
    bt_lv = b3.radio("Nível do VaR", list(BT_LEVELS), format_func=lambda v: f"{v:.1%}".replace(".", ","),
                     horizontal=True)
    rotulo = f"VaR {bt_lv:.1%} · 1 dia · {CVAR_YEARS} anos".replace(".", ",")
    rotulo_es = f"ES {bt_lv:.1%} · 1 dia · {CVAR_YEARS} anos".replace(".", ",")

    s = bt_s[bt_s["setor"] == bt_sel].set_index("data")
    var_col, es_col = f"var_{bt_met}_{bt_lv}", f"es_{bt_met}_{bt_lv}"
    exc = -s["ret"] > s[var_col]
    linha = bt_res[(bt_res["setor"] == bt_sel) & (bt_res["metodo"] == bt_met) & (bt_res["nivel"] == bt_lv)].iloc[0]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Exceções", f"{int(linha['excecoes'])}", f"esperadas {linha['esperadas']:.0f}", delta_color="off")
    m2.metric("Taxa observada", f"{linha['taxa_obs']:.2%}", f"esperada {linha['taxa_esp']:.1%}",
              delta_color="off")
    m3.metric("Kupiec (p-valor)", f"{linha['p_kupiec']:.3f}",
              "rejeita" if linha["p_kupiec"] < BT_SIGNIF else "não rejeita", delta_color="off")
    m4.metric("ES: Z2 (p-valor)", f"{linha['z2']:+.2f}", f"p = {linha['p_z2']:.3f}"
              + (" · rejeita" if linha["p_z2"] < BT_SIGNIF else ""), delta_color="off")
    m5.metric(f"Semáforo ({BT_TL_DAYS} dias)", SEMAFORO[linha["semaforo"]],
              f"{int(linha[f'excecoes_{BT_TL_DAYS}'])} exceções · FRTB {linha['frtb_mesa']}", delta_color="off")

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=s.index, y=s["ret"], mode="lines", name="Retorno diário",
                                line=dict(width=1, color="#64748b")))
    fig_bt.add_trace(go.Scatter(x=s.index, y=-s[var_col], mode="lines", name=f"−{rotulo}",
                                line=dict(width=1.5, color="#f97316")))
    fig_bt.add_trace(go.Scatter(x=s.index, y=-s[es_col], mode="lines", name=f"−{rotulo_es}",
                                line=dict(width=1, color="#a855f7", dash="dot")))
    fig_bt.add_trace(go.Scatter(x=s.index[exc], y=s["ret"][exc], mode="markers", name="Exceção",
                                marker=dict(size=6, color="#ef4444", symbol="x")))
    fig_bt.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor=BG, plot_bgcolor=BG,
                         font=dict(color="#e5e7eb"), hovermode="x unified",
                         legend=dict(orientation="h", y=1.02, yanchor="bottom"),
                         yaxis=dict(tickformat=".0%", gridcolor="#1f2937"),
                         xaxis=dict(gridcolor="#1f2937", rangeslider=dict(visible=True, thickness=0.06)))
    st.plotly_chart(fig_bt, width="stretch")
    st.caption(f"{METODOS[bt_met]} · {rotulo}. O VaR de cada dia é previsto só com dados até à véspera "
               f"(janela móvel de {BT_WINDOW} dias úteis; GARCH re-estimado mensalmente). "
               "Exceção = perda real maior do que o VaR previsto. O ES é a perda média prevista "
               "para os dias de exceção.")

    tab_bt = bt_res.assign(
        Método=bt_res["metodo"].map(METODOS),
        Nível=bt_res["nivel"].map(lambda v: f"{v:.1%}".replace(".", ",")),
        Taxa=bt_res["taxa_obs"].map("{:.2%}".format) + " vs " + bt_res["taxa_esp"].map("{:.1%}".format),
        Semáforo=bt_res["semaforo"].map(SEMAFORO),
        FRTB=bt_res["frtb_mesa"].map({"aprovada": "✅ aprovada", "reprovada": "❌ reprovada"}),
    ).rename(columns={"z2": "Z2 (ES)", "p_z2": "p Z2", "setor": "Setor", "dias": "Dias", "excecoes": "Exceções",
                      "p_kupiec": "p Kupiec", "p_christoffersen": "p Christoffersen",
                      "p_cc": "p condicional", f"excecoes_{BT_TL_DAYS}": f"Exceções {BT_TL_DAYS}d"})
    cols_bt = ["Setor", "Método", "Nível", "Dias", "Exceções", "Taxa", "p Kupiec", "p Christoffersen",
               "p condicional", "Z2 (ES)", "p Z2", f"Exceções {BT_TL_DAYS}d", "Semáforo", "FRTB"]
    so_setor = st.toggle("Mostrar só o setor selecionado", value=False)
    tab_bt = tab_bt[tab_bt["Setor"] == bt_sel] if so_setor else tab_bt

    def _rejeita(v: float) -> str:
        return "color: #ef4444; font-weight: 600" if v < BT_SIGNIF else ""

    st.dataframe(tab_bt[cols_bt].style
                 .format({"p Kupiec": "{:.3f}", "p Christoffersen": "{:.3f}", "p condicional": "{:.3f}",
                          "Z2 (ES)": "{:+.2f}", "p Z2": "{:.3f}"})
                 .map(_rejeita, subset=["p Kupiec", "p Christoffersen", "p condicional", "p Z2"]),
                 hide_index=True, width="stretch")
    st.caption(f"A vermelho: p-valor < {BT_SIGNIF:.0%} (rejeita H0). Kupiec testa a taxa de exceções; "
               "Christoffersen testa se as exceções aparecem em grupos; o condicional testa as duas coisas; "
               "Z2 (Acerbi–Székely) testa o ES (negativo = ES subestimado; p-valor por simulação). "
               f"Semáforo de Basileia nos últimos {BT_TL_DAYS} dias (a 99%: verde 0–4, amarelo 5–9, "
               "vermelho 10+; a 97,5% usa as mesmas probabilidades binomiais). FRTB por mesa: reprovada com mais "
               "de 12 exceções a 99% ou 30 a 97,5%. Metodologia: "
               "docs/nota-metodologica.md.")

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
