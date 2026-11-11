"""Cockpit de Risco Setorial Interativo.

Correr a partir da raiz do projeto:
    .venv/bin/streamlit run app/cockpit/app.py
Sem rede, ou com o Yahoo a limitar pedidos, usa o snapshot versionado em app/cockpit/snapshot/
(COCKPIT_SNAPSHOT=1 força esse modo, como no Streamlit Community Cloud).
Metodologia: docs/nota-metodologica.md
"""
import json
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
                             MODEL_RISK_LEVEL, MODEL_RISK_MULT, OBSIDIAN_FOLDER, OBSIDIAN_VAULT,
                             OFFICIAL_VAR_METHOD, PANIC_Q, PORTFOLIO_WEIGHTS, RATE_LOOKBACK,
                             RATE_SERIES, SCORE_WEIGHTS, SECTOR_PROFILE, SECTORS, SNAPSHOT)
from src.models import cockpit as ck  # noqa: E402
from src.models.scorecard import COMPONENTS  # noqa: E402

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
# Ao vivo (Yahoo + FRED) por omissão; snapshot versionado se falharem ou com COCKPIT_SNAPSHOT=1.
FORCAR_SNAPSHOT = os.environ.get("COCKPIT_SNAPSHOT", "").lower() in ("1", "true", "sim")
FONTES: set[str] = set()          # "ao vivo" e/ou "snapshot", para o aviso no cabeçalho


@st.cache_data(show_spinner=False)
def snap_precos() -> pd.DataFrame:
    return pd.read_parquet(SNAPSHOT / "prices.parquet").astype(float)


def precos_snapshot(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    p = snap_precos()
    falta = [t for t in tickers if t not in p]
    if falta:
        raise RuntimeError(f"snapshot sem {falta}")
    anos = int(period.rstrip("y"))
    return p.loc[p.index > p.index.max() - pd.DateOffset(years=anos), list(tickers)]


@st.cache_data(ttl=3600, show_spinner="🤖 A descarregar preços dos setores…")
def precos_vivo(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return ck.download_close(list(tickers), period)


def precos(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    if not (FORCAR_SNAPSHOT or st.session_state.get("yahoo_falhou")):
        try:
            out = precos_vivo(tickers, period)
            FONTES.add("ao vivo")
            return out
        except Exception:  # noqa: BLE001  Yahoo em baixo ou a limitar: snapshot até ao fim da sessão
            st.session_state["yahoo_falhou"] = True
    FONTES.add("snapshot")
    return precos_snapshot(tickers, period)


@st.cache_data(ttl=3600, show_spinner="🏦 A ler o Treasury 10Y no FRED…")
def juros_vivo(start: str) -> pd.Series:
    return ck.fetch_rates(RATE_SERIES, start)


def juros(start: str) -> pd.Series:
    if not (FORCAR_SNAPSHOT or st.session_state.get("fred_falhou")):
        try:
            return juros_vivo(start)
        except Exception:  # noqa: BLE001
            st.session_state["fred_falhou"] = True
    y = pd.read_parquet(SNAPSHOT / "rates.parquet")["DGS10"]
    return y[y.index >= pd.Timestamp(start)]


# Resultados do pipeline: data/features se existir (local), senão o snapshot (Cloud).
FEAT = FEATURES if (FEATURES / "scorecard.csv").exists() and not FORCAR_SNAPSHOT else SNAPSHOT
SNAP_META = json.loads((SNAPSHOT / "meta.json").read_text()) if (SNAPSHOT / "meta.json").exists() else {}


setores = list(SECTORS)
try:
    rets_all = ck.returns(precos(tuple(setores + [MARKET]), periodo))
except Exception as e:  # noqa: BLE001
    st.error(f"Sem preços: o Yahoo falhou e o snapshot não está disponível ({e}).")
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
nivel = f"{alpha:.1%}".replace(".", ",")        # ex.: 97,5% (nunca arredondar o nível do rótulo)
st.title("🛡️ Cockpit de Risco Setorial")
st.caption(f"Dados de {rets.index[0].date()} a {rets.index[-1].date()} · CVaR {nivel} · 1 dia · "
           f"pânico = {panic_q:.0%} piores dias do SPY ({int((mkt <= mkt.quantile(panic_q)).sum())} dias)")
if "snapshot" in FONTES:
    st.info(f"📦 Modo snapshot: preços e juros guardados no repositório (dados até "
            f"{SNAP_META.get('dados_ate', '?')}), porque {'COCKPIT_SNAPSHOT está ativo' if FORCAR_SNAPSHOT else 'o Yahoo não respondeu'}. "
            "Os resultados do pipeline (scorecard, backtest, carteira) são os da mesma data.")
elif FEAT == SNAPSHOT:
    st.caption(f"Resultados do pipeline: snapshot de {SNAP_META.get('dados_ate', '?')}.")

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

# ============================================================================ scorecard
st.subheader("🎯 Scorecard de risco por setor")
sc_csv, off_csv = FEAT / "scorecard.csv", FEAT / "official_risk.csv"


@st.cache_data(show_spinner=False)
def scorecard_dados(mtime: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(sc_csv, index_col=0), pd.read_csv(off_csv)


if not (sc_csv.exists() and off_csv.exists()):
    st.info("Ainda não há scorecard. Corre `python run_pipeline.py --offline`.")
else:
    sc, off = scorecard_dados(sc_csv.stat().st_mtime)
    data_prev = off["data_previsao"].iloc[0]
    SEM = {"verde": "🟢 verde", "amarelo": "🟡 amarelo", "vermelho": "🔴 vermelho"}
    NOMES = {"cauda": "Cauda", "sistemico": "Sistémico", "stress": "Stress", "regime": "Regime", "momento": "Momento"}
    CORES_C = {"cauda": "#f97316", "sistemico": "#ef4444", "stress": "#a855f7", "regime": "#3b82f6", "momento": "#14b8a6"}

    s1, s2, s3, s4 = st.columns(4)
    lider = sc.index[0]
    s1.metric("Setor de maior risco", lider, f"score {sc.loc[lider, 'score_final']:.0f}/100", delta_color="off")
    s2.metric(f"ES 97,5% oficial · {data_prev}", f"{sc.loc[lider, 'bruto_cauda']:.2%}".replace(".", ","),
              f"{lider} · modelo {OFFICIAL_VAR_METHOD.upper()}", delta_color="off")
    flag = sc[sc["risco_modelo"].astype(bool)]
    s3.metric("Risco de modelo", f"{len(flag)} setor(es)", ", ".join(flag.index) or "nenhum", delta_color="off")
    s4.metric("Críticos (P × I ≥ 15)", int((sc["impacto"] * sc["probabilidade"] >= 15).sum()), delta_color="off")

    col_g, col_t = st.columns([2, 3])
    with col_g:
        ordem = sc.index[::-1]                       # o maior score fica no topo do gráfico
        fig_sc = go.Figure()
        for k in COMPONENTS:                          # contribuição de cada componente: 100 · w_k · c_k (soma = score)
            fig_sc.add_trace(go.Bar(y=ordem, x=100 * SCORE_WEIGHTS[k] * sc.loc[ordem, k], orientation="h",
                                    name=f"{NOMES[k]} ({SCORE_WEIGHTS[k]:.0%})", marker_color=CORES_C[k]))
        extra = (sc.loc[ordem, "score_final"] - sc.loc[ordem, "score"]).clip(lower=0)
        fig_sc.add_trace(go.Bar(y=ordem, x=extra, orientation="h", name="Ajuste de risco de modelo",
                                marker=dict(color="rgba(234,179,8,0.35)", pattern_shape="/",
                                            line=dict(color="#eab308", width=1))))
        fig_sc.update_layout(barmode="stack", height=420, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor=BG,
                             plot_bgcolor=BG, font=dict(color="#e5e7eb"),
                             title=dict(text="Score 0–100 por componente", x=0),
                             xaxis=dict(range=[0, 100], gridcolor="#1f2937"),
                             legend=dict(orientation="h", y=-0.12, font=dict(size=11)))
        st.plotly_chart(fig_sc, width="stretch")
    with col_t:
        tab_sc = pd.DataFrame({
            "#": sc["rank"],
            "Setor": [f"{SECTORS[t]} ({t})" for t in sc.index],
            "Score": sc["score_final"],
            "P × I": [f"P{p} × I{i} = {p * i}" for p, i in zip(sc["probabilidade"], sc["impacto"])],
            "Semáforo": sc["semaforo"].map(SEM) + sc["risco_modelo"].map({True: " ⚠", False: ""}),
            "ES 97,5%": sc["bruto_cauda"],
            **{NOMES[k]: sc[k] for k in COMPONENTS},
        })
        st.dataframe(tab_sc, hide_index=True, width="stretch", height=420, column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
            "ES 97,5%": st.column_config.NumberColumn(format="percent"),
            **{NOMES[k]: st.column_config.ProgressColumn(NOMES[k], min_value=0, max_value=1, format="%.2f")
               for k in COMPONENTS},
        })
    mult = " · ".join(f"{c} ×{m:.2f}".replace(".", ",") for c, m in MODEL_RISK_MULT.items())
    st.caption(f"Score = 100 · Σ peso × componente; cada componente vai de 0 (o mais baixo dos 11) a 1 (o mais alto). "
               f"Cauda = ES 97,5% previsto pelo {OFFICIAL_VAR_METHOD.upper()} para {data_prev}. "
               f"Risco de modelo: semáforo do backtest oficial a {MODEL_RISK_LEVEL:.0%} ({mult}). "
               "P × I segue a mesma regra do mapa de riscos. Metodologia: docs/nota-metodologica.md, secção 8.")

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
        st.caption(f"CVaR {nivel} do ETF {sel} no mesmo período: {pai:.2%}.")
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
bt_csv, bt_ser = FEAT / "var_backtest.csv", FEAT / "var_backtest_series.parquet"


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

# ============================================================================ visão conjunta
st.markdown("---")
st.subheader("🧩 Visão conjunta: carteira dos 11 setores")
rc_csv, stt_csv, lw_csv = (FEAT / "risk_contribution.csv", FEAT / "stress_tests.csv",
                           FEAT / "covariance_shrinkage.csv")


@st.cache_data(show_spinner=False)
def conjunta_dados(mtime: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(rc_csv, index_col=0), pd.read_csv(stt_csv), pd.read_csv(lw_csv)


if not all(f.exists() for f in (rc_csv, stt_csv, lw_csv)):
    st.info("Ainda não há resultados da visão conjunta. Corre `python run_pipeline.py --offline`.")
else:
    rc, stt, lw = conjunta_dados(rc_csv.stat().st_mtime)
    cart, rc_s = rc.loc["Carteira"], rc.drop(index="Carteira")
    rot_cvar = f"CVaR {CVAR_LEVEL:.1%} · 1 dia · {CVAR_YEARS} anos".replace(".", ",")
    pesos_txt = "pesos iguais" if PORTFOLIO_WEIGHTS is None else "pesos de PORTFOLIO_WEIGHTS"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Volatilidade · {CVAR_YEARS} anos", f"{cart['contrib_vol']:.2%}",
              f"diversificação {(rc_s['peso'] * rc_s['vol_individual']).sum() / cart['contrib_vol']:.2f}×",
              delta_color="off")
    k2.metric(rot_cvar, f"{cart['contrib_cvar']:.2%}",
              f"diversificação {(rc_s['peso'] * rc_s['cvar_individual']).sum() / cart['contrib_cvar']:.2f}×",
              delta_color="off")
    top_rc = rc_s["pct_cvar"].idxmax()
    k3.metric("Maior contribuição para o CVaR", top_rc, f"{rc_s.loc[top_rc, 'pct_cvar']:.1%} do total",
              delta_color="off")
    pior = stt[stt["serie"] == "Carteira"].sort_values("perda").iloc[-1]
    k4.metric("Pior stress histórico", f"−{pior['perda']:.1%}", pior["cenario"], delta_color="off")

    col_rc, col_st = st.columns(2)
    with col_rc:
        fig_rc = go.Figure()
        fig_rc.add_trace(go.Bar(x=rc_s.index, y=rc_s["pct_vol"], name="% da volatilidade",
                                marker_color="#64748b"))
        fig_rc.add_trace(go.Bar(x=rc_s.index, y=rc_s["pct_cvar"], name=f"% do {rot_cvar}",
                                marker_color="#f97316"))
        fig_rc.add_trace(go.Scatter(x=rc_s.index, y=rc_s["peso"], mode="lines", name="Peso",
                                    line=dict(color="#e5e7eb", dash="dot", width=1)))
        fig_rc.update_layout(barmode="group", height=380, margin=dict(l=0, r=0, t=30, b=0),
                             paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#e5e7eb"),
                             title=dict(text="Contribuição para o risco (Euler)", x=0),
                             legend=dict(orientation="h", y=-0.12), yaxis=dict(tickformat=".0%", gridcolor="#1f2937"))
        st.plotly_chart(fig_rc, width="stretch")
    with col_st:
        ordem = setores + ["Carteira", MARKET]
        piv = stt.pivot(index="serie", columns="cenario", values="perda").reindex(ordem)
        prx = stt.pivot(index="serie", columns="cenario", values="proxy").reindex(ordem).fillna(False)
        txt = piv.map(lambda v: f"{-v:.0%}") + prx.map(lambda p: "*" if p else "")
        hm_st = px.imshow(-piv, color_continuous_scale="RdYlGn", color_continuous_midpoint=0, aspect="auto",
                          labels=dict(color="Retorno"))
        hm_st.update_traces(text=txt.values, texttemplate="%{text}")
        hm_st.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor=BG,
                            font=dict(color="#e5e7eb"), coloraxis_showscale=False,
                            title=dict(text="Stress tests históricos (pico → vale do SPY)", x=0))
        st.plotly_chart(hm_st, width="stretch")
    st.caption(f"Carteira de referência com {pesos_txt}, re-balanceada só no início de cada episódio "
               f"(buy-and-hold). Contribuições de Euler: somam a volatilidade (covariância Ledoit-Wolf) e o "
               f"{rot_cvar} da carteira. * = setor ainda sem cotação no episódio: proxy β × SPY com o β atual.")

    cart_st = stt[stt["serie"] == "Carteira"].set_index("cenario")
    contrib = stt[~stt["serie"].isin(["Carteira", MARKET])]
    maior = contrib.loc[contrib.groupby("cenario")["contrib"].idxmax()].set_index("cenario")
    spy_st = stt[stt["serie"] == MARKET].set_index("cenario")
    st.dataframe(pd.DataFrame({
        "Período": cart_st["inicio"] + " → " + cart_st["fim"],
        "Perda da carteira": cart_st["perda"].map("{:.1%}".format),
        f"Perda do {MARKET}": spy_st["perda"].map("{:.1%}".format),
        "Drawdown máximo": cart_st["max_dd"].map("{:.1%}".format),
        "Pior dia": cart_st["pior_dia"].map("{:.2%}".format),
        "Pior dia / CVaR atual": cart_st["pior_dia_em_cvar"].map("{:.1f}×".format),
        "Maior contribuinte": maior["serie"] + " (" + maior["contrib"].map("{:.1%}".format) + ")",
    }).rename_axis("Cenário").reset_index(), hide_index=True, width="stretch")

    with st.expander("📐 Covariância: Ledoit-Wolf vs amostral"):
        st.dataframe(pd.DataFrame({
            "Janela (dias)": lw["janela"],
            "T / N": lw["t_sobre_n"].map("{:.1f}".format),
            "Encolhimento δ": lw["delta"].map("{:.3f}".format),
            "Nº condição amostral": lw["cond_amostral"].map("{:.1f}".format),
            "Nº condição Ledoit-Wolf": lw["cond_ledoit_wolf"].map("{:.1f}".format),
            "Vol. fora da amostra (amostral)": lw["vol_oos_amostral"].map("{:.2%}".format),
            "Vol. fora da amostra (LW)": lw["vol_oos_ledoit_wolf"].map("{:.2%}".format),
            "Redução": lw["reducao_vol"].map("{:.1%}".format),
            "Desde": lw["oos_inicio"],
        }), hide_index=True, width="stretch")
        st.caption("Carteira de variância mínima sem restrições, re-estimada mensalmente só com dados passados. "
                   "Quanto menor T/N, mais o encolhimento ajuda.")

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
if not vault.is_dir():                       # ex.: Streamlit Cloud, sem cofre local
    c_a.caption("Cofre não encontrado neste computador: usa o botão para descarregar a nota.")
elif c_a.button("💾 Exportar estado de hoje", type="primary"):
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(nota, encoding="utf-8")
    st.success(f"Gravado em {destino}")
c_b.download_button("⬇️ Descarregar .md", nota, file_name=destino.name, mime="text/markdown")
with st.expander("Pré-visualizar nota"):
    st.code(nota, language="markdown")
