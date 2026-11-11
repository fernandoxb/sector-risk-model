"""Cockpit de Risco Setorial Interativo.

Correr a partir da raiz do projeto:
    .venv/bin/streamlit run app/cockpit/app.py
Sem rede, ou com o Yahoo a limitar pedidos, usa o snapshot versionado em app/cockpit/snapshot/
(COCKPIT_SNAPSHOT=1 força esse modo, como no Streamlit Community Cloud).
Separadores: Resumo · Setores · Ficha do setor · Análise ao vivo · Fatores macro · Modelo (backtest) ·
Carteira e stress · Glossário. Tema em .streamlit/config.toml.
Metodologia: docs/nota-metodologica.md
"""
import json
import os
import sys
from datetime import date
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.settings import (BT_LEVELS, BT_SIGNIF, BT_TL_DAYS, BT_WINDOW,  # noqa: E402
                             CVAR_LEVEL, CVAR_YEARS, FEATURES, GROWTH_SECTORS, LINK_T_MIN, MACRO_FACTORS,
                             MARKET, MODEL_RISK_LEVEL, MODEL_RISK_MULT, MOMENTUM_DAYS, OBSIDIAN_FOLDER,
                             OBSIDIAN_VAULT, OFFICIAL_VAR_METHOD, PANIC_Q, PI_LEVELS, PORTFOLIO_WEIGHTS,
                             RATE_LOOKBACK, RATE_SERIES, RATE_THRESHOLD_PP, SCORE_WEIGHTS, SECTOR_PROFILE,
                             SECTORS, SHOCK_DIFF, SHOCK_LOGDIFF, SHOCK_MARKET, SNAPSHOT)
from src.models import cockpit as ck  # noqa: E402
from src.models.portfolio import shock_returns  # noqa: E402
from src.models.scorecard import COMPONENTS  # noqa: E402
from src.report import formato as fm  # noqa: E402
from src.report.alerts import pi_level, pi_min  # noqa: E402

st.set_page_config(page_title="Cockpit de Risco Setorial", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

# ============================================================================ estilo comum
BG = "rgba(0,0,0,0)"          # transparente: os gráficos herdam o fundo do tema (.streamlit/config.toml)
GRID = "#1e293b"
TXT = "#e2e8f0"
MUTED = "#64748b"
# verde/amarelo/vermelho ficam reservados ao risco; o tipo de setor usa outra paleta
RISCO = {"verde": "#22c55e", "amarelo": "#eab308", "vermelho": "#ef4444"}
SEM = {"verde": "🟢 verde", "amarelo": "🟡 amarelo", "vermelho": "🔴 vermelho"}
SEV = {"alta": ("red", "alta"), "média": ("orange", "média"), "baixa": ("gray", "baixa")}
NOMES = {"cauda": "Cauda", "sistemico": "Sistémico", "stress": "Stress", "regime": "Regime", "momento": "Momento"}
CORES_C = {"cauda": "#6366f1", "sistemico": "#0ea5e9", "stress": "#a855f7", "regime": "#14b8a6", "momento": "#ec4899"}
METODOS = {"historico": "Histórico", "garch": "GJR-GARCH(1,1)-t", "fhs": "GARCH-FHS"}
OFICIAL = OFFICIAL_VAR_METHOD.upper()
CRIT = PI_LEVELS[-1][1]
CRIT_MIN = pi_min(CRIT)
N_CVAR = fm.nivel(CVAR_LEVEL)


def layout(fig: go.Figure, height: int, title: str | None = None, **kw) -> go.Figure:
    """Tema escuro comum e números com vírgula decimal (separators="," + espaço)."""
    kw.setdefault("margin", dict(l=0, r=0, t=34 if title else 8, b=0))
    fig.update_layout(height=height, paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TXT), separators=", ",
                      title=dict(text=title, x=0) if title else None, **kw)
    return fig


def metrica(col, rotulo: str, valor: str, nota: str | None = None, ajuda: str | None = None):
    """Indicador com nota descritiva (sem seta: a nota não é uma variação)."""
    col.metric(rotulo, valor, nota, delta_color="off", delta_arrow="off", help=ajuda)


def nome(t: str) -> str:
    return f"{SECTORS[t]} ({t})"


def seletor_periodo(fig: go.Figure, anos: int = 3) -> None:
    """Botões 1A/3A/5A/Tudo e janela inicial dos últimos `anos` (sem a barra de zoom, que duplica as séries)."""
    fig.update_xaxes(rangeselector=dict(
        buttons=[dict(count=1, label="1A", step="year", stepmode="backward"),
                 dict(count=3, label="3A", step="year", stepmode="backward"),
                 dict(count=5, label="5A", step="year", stepmode="backward"),
                 dict(step="all", label="Tudo")],
        bgcolor="#131c2e", activecolor="#3b82f6", font=dict(color=TXT), x=0, y=1.02))
    x = [d for tr in fig.data for d in (tr.x if tr.x is not None else [])]
    if x:
        fim = pd.Timestamp(max(x))
        ini = fim - pd.DateOffset(years=anos)
        fig.update_xaxes(range=[ini, fim])
        # escala vertical ajustada à janela inicial (senão os picos de 2008/2020 achatam o gráfico)
        ys = [y for tr in fig.data if tr.x is not None and tr.y is not None
              for xx, y in zip(tr.x, tr.y) if pd.Timestamp(xx) >= ini and pd.notna(y)]
        if ys:
            lo, hi = min(ys), max(ys)
            pad = 0.08 * (hi - lo)
            fig.update_yaxes(range=[min(lo - pad, 0), hi + pad])


# ============================================================================ dados (cache)
# Ao vivo (Yahoo + FRED) por omissão; snapshot versionado se falharem ou com COCKPIT_SNAPSHOT=1.
FORCAR_SNAPSHOT = os.environ.get("COCKPIT_SNAPSHOT", "").lower() in ("1", "true", "sim")
FONTES: set[str] = set()          # "ao vivo" e/ou "snapshot"


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


@st.cache_data(ttl=3600, show_spinner="🤖 A descarregar preços…")
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


@st.cache_data(show_spinner=False)
def ler_csv(path: str, mtime: float, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw)


def feat_csv(nome_f: str, **kw) -> pd.DataFrame | None:
    f = FEAT / nome_f
    return ler_csv(str(f), f.stat().st_mtime, **kw) if f.exists() else None


@st.cache_data(show_spinner=False)
def ler_series(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


sc = feat_csv("scorecard.csv", index_col=0)
off = feat_csv("official_risk.csv", index_col="setor")
bt_res = feat_csv("var_backtest.csv")
rc = feat_csv("risk_contribution.csv", index_col=0)
stt = feat_csv("stress_tests.csv")
lw = feat_csv("covariance_shrinkage.csv")
links = feat_csv("factor_links.csv")
metr = feat_csv("sector_metrics.csv", index_col=0)
hist = feat_csv("score_history.csv")
BT_SER = FEAT / "var_backtest_series.parquet"
alertas = json.loads((FEAT / "alerts.json").read_text()) if (FEAT / "alerts.json").exists() else None
PIPE_ASOF = str(off["dados_ate"].iloc[0]) if off is not None else "—"
PREV_DATA = str(off["data_previsao"].iloc[0]) if off is not None else "—"

# ============================================================================ barra lateral (só o que é global)
st.sidebar.title("🛡️ Risco Setorial")
st.sidebar.caption("11 ETFs setoriais SPDR face ao SPY. Os parâmetros de cada análise estão no próprio separador.")
janela = st.sidebar.selectbox("📅 Janela histórica (análise ao vivo)", ["1 Ano", "2 Anos", "3 Anos"],
                              index=min(max(CVAR_YEARS, 1), 3) - 1,
                              help="Afeta só o separador Análise ao vivo. Os resultados do pipeline usam sempre "
                                   f"{CVAR_YEARS} anos.")
periodo = {"1 Ano": "1y", "2 Anos": "2y", "3 Anos": "3y"}[janela]

setores = list(SECTORS)
try:
    rets_all = ck.returns(precos(tuple(setores + [MARKET]), periodo))
except Exception as e:  # noqa: BLE001
    st.error(f"Sem preços: o Yahoo falhou e o snapshot não está disponível ({e}).")
    st.stop()
rets, mkt = rets_all[setores], rets_all[MARKET]
grafo, corr = ck.mst(rets)

regime, d_yield = None, None
try:
    y = juros((rets.index[0] - pd.Timedelta(days=200)).date().isoformat())
    regime = ck.rate_regime(y, RATE_LOOKBACK, RATE_THRESHOLD_PP)
    d_yield = y.diff().reindex(rets.index)
except Exception:  # noqa: BLE001
    pass

st.sidebar.markdown("**Datas dos dados**")
st.sidebar.markdown(
    f"- Pipeline (scorecard, backtest, carteira): até **{PIPE_ASOF}**\n"
    f"- Previsão oficial ({OFICIAL}) para: **{PREV_DATA}**\n"
    f"- Preços da análise ao vivo: até **{rets.index[-1].date()}**\n"
    f"- Treasury 10Y: até **{regime['asof'] if regime else '—'}**")
fonte = ("snapshot do repositório" if "snapshot" in FONTES else "Yahoo Finance (ao vivo)")
st.sidebar.caption(f"Preços: {fonte}. O FRED publica os juros com 1 dia de atraso; a previsão oficial é "
                   "para o dia útil seguinte ao último fecho.")
st.sidebar.markdown("---")
st.sidebar.markdown("[Código no GitHub](https://github.com/fernandoxb/sector-risk-model) · "
                    "[Nota metodológica (PDF)](https://github.com/fernandoxb/sector-risk-model/blob/main/docs/"
                    "nota-metodologica.pdf)")

# ============================================================================ cabeçalho
st.title("🛡️ Cockpit de Risco Setorial")
if "snapshot" in FONTES:
    st.info(f"📦 Modo snapshot: preços e juros guardados no repositório (até {SNAP_META.get('dados_ate', '?')}), "
            f"porque {'COCKPIT_SNAPSHOT está ativo' if FORCAR_SNAPSHOT else 'o Yahoo não respondeu'}.")

tabs = st.tabs(["📋 Resumo", "🎯 Setores", "🔎 Ficha do setor", "📊 Análise ao vivo", "🔗 Fatores macro",
                "🧪 Modelo (backtest)", "🧩 Carteira e stress", "📖 Glossário"])
T_RESUMO, T_SETORES, T_FICHA, T_VIVO, T_FATORES, T_MODELO, T_CARTEIRA, T_GLOSSARIO = tabs

# ============================================================================ 1. Resumo
with T_RESUMO:
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

# ============================================================================ 2. Setores (scorecard)
with T_SETORES:
    if sc is None:
        st.info("Ainda não há scorecard. Corre `python run_pipeline.py --offline`.")
    else:
        st.subheader("Scorecard de risco por setor (0–100)")
        st.caption("Junta cinco medidas de risco num número: 100 = o setor mais arriscado em todas, 0 = o menos "
                   f"arriscado em todas. Dados até {PIPE_ASOF}.")
        s1, s2, s3 = st.columns(3)
        lider = sc.index[0]
        metrica(s1, "Setor de maior risco", lider, f"score {fm.num(sc.loc[lider, 'score_final'], 0)}/100")
        crit = sc[sc["impacto"] * sc["probabilidade"] >= CRIT_MIN]
        metrica(s2, f"Setores em nível {CRIT}", str(len(crit)), ", ".join(crit.index) or "nenhum",
                ajuda=f"Nível {CRIT}: probabilidade × impacto (P × I) ≥ {CRIT_MIN}, numa escala de 1 a 25. "
                      "A mesma regra dos alertas do Sentinela.")
        flag = sc[sc["risco_modelo"].astype(bool)]
        metrica(s3, "Setores com risco de modelo", str(len(flag)), ", ".join(flag.index) or "nenhum",
                ajuda=f"O backtest do {OFICIAL} a {fm.nivel(MODEL_RISK_LEVEL)} não está verde nos últimos "
                      f"{BT_TL_DAYS} dias: o score é agravado ({', '.join(f'{c} ×{fm.num(m)}' for c, m in MODEL_RISK_MULT.items())}).")

        col_g, col_t = st.columns([5, 6])
        with col_g:
            ordem = sc.index[::-1]
            fig_sc = go.Figure()
            for k in COMPONENTS:
                fig_sc.add_trace(go.Bar(y=ordem, x=100 * SCORE_WEIGHTS[k] * sc.loc[ordem, k], orientation="h",
                                        name=f"{NOMES[k]} ({fm.pct(SCORE_WEIGHTS[k], 0)})", marker_color=CORES_C[k],
                                        hovertemplate="%{y}: %{x:.1f} pontos<extra>" + NOMES[k] + "</extra>"))
            extra = (sc.loc[ordem, "score_final"] - sc.loc[ordem, "score"]).clip(lower=0)
            fig_sc.add_trace(go.Bar(y=ordem, x=extra, orientation="h", name="Risco de modelo",
                                    marker=dict(color="rgba(234,179,8,0.35)", pattern_shape="/",
                                                line=dict(color="#eab308", width=1))))
            layout(fig_sc, 470, "Pontos de cada componente", barmode="stack",
                   xaxis=dict(range=[0, 100], gridcolor=GRID), margin=dict(l=0, r=0, t=90, b=0),
                   legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=11),
                               entrywidth=0.33, entrywidthmode="fraction", traceorder="normal"))
            st.plotly_chart(fig_sc, width="stretch")
        with col_t:
            st.dataframe(pd.DataFrame({
                "Setor": [f"{int(r)}. {t}" for t, r in zip(sc.index, sc["rank"])],
                "Score": sc["score_final"].round(0).astype(int),
                "P × I": [f"{p * i} · {pi_level(p, i)}" for p, i in zip(sc["probabilidade"], sc["impacto"])],
                "Modelo": sc["semaforo"].map(lambda v: SEM[v].split()[0]) + sc["risco_modelo"].map({True: " ⚠", False: ""}),
                "ES amanhã": sc["bruto_cauda"].map(fm.pct),
            }), hide_index=True, width="stretch", height=470, column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                "P × I": st.column_config.TextColumn("P × I", help="Probabilidade (1–5) × impacto (1–5) e nível."),
                "Modelo": st.column_config.TextColumn("Modelo", help="Semáforo do backtest oficial; ⚠ = risco de modelo."),
                "ES amanhã": st.column_config.TextColumn(f"ES {N_CVAR}", help=f"ES {N_CVAR} previsto pelo {OFICIAL}."),
            })
        with st.expander("Como se calcula o score"):
            st.markdown(
                "Cada componente é ordenado entre os 11 setores e vai de 0 (o mais baixo) a 1 (o mais alto); "
                "o score é 100 × a soma ponderada.\n\n"
                f"| Componente | Peso | O que mede |\n|---|---|---|\n"
                f"| Cauda | {fm.pct(SCORE_WEIGHTS['cauda'], 0)} | ES {N_CVAR} previsto pelo {OFICIAL} para amanhã |\n"
                f"| Sistémico | {fm.pct(SCORE_WEIGHTS['sistemico'], 0)} | quanto o setor pesa no CVaR da carteira, por unidade de peso (Euler) |\n"
                f"| Stress | {fm.pct(SCORE_WEIGHTS['stress'], 0)} | perda média nos cenários 2008, COVID-19 e 2022 |\n"
                f"| Regime | {fm.pct(SCORE_WEIGHTS['regime'], 0)} | volatilidade dos últimos 63 dias face à história do próprio setor |\n"
                f"| Momento | {fm.pct(SCORE_WEIGHTS['momento'], 0)} | exceções ao VaR {N_CVAR} oficial nos últimos {MOMENTUM_DAYS} dias |\n\n"
                "**P × I**: impacto (I) = quintil de cauda + sistémico + stress; probabilidade (P) = quintil de "
                "regime + momento. Níveis: "
                + ", ".join(f"{n} ≤ {lim}" for lim, n in PI_LEVELS) + ". Detalhes na nota metodológica, secção 8.")

        st.subheader("Evolução do score")
        if hist is None or hist["data"].nunique() < 2:
            st.caption("O histórico do score começa a acumular-se com os runs diários do pipeline (bot Sentinela).")
        else:
            fig_h = go.Figure()
            for t in sc.index:
                h = hist[hist["setor"] == t]
                fig_h.add_trace(go.Scatter(x=pd.to_datetime(h["data"]), y=h["score"], mode="lines+markers", name=t,
                                           hovertemplate=t + " · %{x|%d/%m}: %{y:.1f}<extra></extra>"))
            layout(fig_h, 320, yaxis=dict(range=[0, 100], gridcolor=GRID, title="Score"),
                   xaxis=dict(gridcolor=GRID, tickformat="%d/%m"), legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_h, width="stretch")
            st.caption(f"{hist['data'].nunique()} dias com estado gravado ({hist['data'].min()} → {hist['data'].max()}). "
                       "Um ponto por run diário do pipeline.")

# ============================================================================ 3. Ficha do setor
with T_FICHA:
    if sc is None or off is None or metr is None:
        st.info("Ainda não há resultados do pipeline. Corre `python run_pipeline.py --offline`.")
    else:
        fs = st.selectbox("Setor", setores, index=setores.index(sc.index[0]), format_func=nome, key="ficha")
        r = sc.loc[fs]
        p_, i_ = int(r["probabilidade"]), int(r["impacto"])
        st.caption(f"{SECTOR_PROFILE[fs]['tipo']} · maiores posições: {', '.join(SECTOR_PROFILE[fs]['acoes'])} · "
                   f"resultados do pipeline até {PIPE_ASOF}")
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
                                f"R² {fm.pct(l['r2'], 0)})")
            viz = sorted(((v, grafo[fs][v]["corr"]) for v in grafo.neighbors(fs)), key=lambda x: -x[1])
            st.markdown("**Contágio (vizinhos na MST)**  \n" + " · ".join(f"{v} (r = {fm.num(c)})" for v, c in viz))
            if stt is not None:
                st.markdown("**Nas crises históricas**")
                ss = stt[stt["serie"] == fs].set_index("cenario")
                for cen, l in ss.iterrows():
                    st.markdown(f"- {cen}: **{fm.pct(-l['perda'])}**" + (" (proxy β × SPY)" if l["proxy"] else ""))

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

# ============================================================================ 4. Análise ao vivo
with T_VIVO:
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

    st.subheader("Espinha de contágio (Minimum Spanning Tree)")
    st.caption("Liga os 11 setores pelas 10 correlações mais fortes que formam uma árvore. Setores próximos caem "
               "juntos. **Cor = score de risco** (verde baixo, vermelho alto); **tamanho = CVaR histórico**.")
    pos = nx.kamada_kawai_layout(grafo, weight="weight")
    edge_x, edge_y, mid_x, mid_y, mid_txt = [], [], [], [], []
    for a, b, d in grafo.edges(data=True):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        mid_txt.append(f"{a} ↔ {b}<br>correlação {fm.num(d['corr'])}")
    nos = list(grafo.nodes)
    score_no = [float(sc.loc[n, "score_final"]) if sc is not None else 50.0 for n in nos]
    cmax = tab["cvar_aj"].max()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none",
                             line=dict(width=2, color="#4b5563"), showlegend=False))
    fig.add_trace(go.Scatter(x=mid_x, y=mid_y, mode="markers", hoverinfo="text", hovertext=mid_txt,
                             marker=dict(size=12, color="rgba(0,0,0,0)"), showlegend=False))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nos], y=[pos[n][1] for n in nos], mode="markers+text", text=nos,
        textposition="bottom center", textfont=dict(color=TXT), showlegend=False, hoverinfo="text",
        marker=dict(size=[20 + 45 * tab.loc[n, "cvar_aj"] / cmax for n in nos], color=score_no,
                    colorscale=[[0, RISCO["verde"]], [0.5, RISCO["amarelo"]], [1, RISCO["vermelho"]]],
                    cmin=0, cmax=100, line=dict(width=2, color="#ffffff"),
                    colorbar=dict(title="Score", thickness=12, len=0.6)),
        hovertext=[f"<b>{nome(n)}</b> · {SECTOR_PROFILE[n]['tipo'].lower()}<br>"
                   f"Score {fm.num(s, 0)}/100<br>CVaR {nivel_live}: {fm.pct(tab.loc[n, 'cvar'], 2)}"
                   + (f" (ajustado: {fm.pct(tab.loc[n, 'cvar_aj'], 2)})" if tab.loc[n, "cvar_aj"] != tab.loc[n, "cvar"]
                      else "")
                   + f"<br>β↓ {fm.num(tab.loc[n, 'downside_beta'])} · ligações {grafo.degree(n)}"
                   for n, s in zip(nos, score_no)]))
    layout(fig, 500, hovermode="closest", xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
    st.plotly_chart(fig, width="stretch")

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

    st.subheader("🔍 Microscópio de ações")
    sel = st.selectbox("Setor", setores, format_func=nome)
    acoes = SECTOR_PROFILE[sel]["acoes"]
    drill = pd.DataFrame()
    try:
        r_ac = ck.returns(precos(tuple(acoes + [sel, MARKET]), periodo))
        t_ac = ck.tail_table(r_ac[acoes], r_ac[MARKET], alpha, panic_q)
        pai = ck.cvar(r_ac[sel], alpha)
        drill = pd.DataFrame({
            "Ticker": acoes,
            f"CVaR {nivel_live}": t_ac["cvar"].map(lambda v: fm.pct(v, 2)).values,
            "Downside beta": t_ac["downside_beta"].map(fm.num).values,
            "CVaR face ao ETF": (t_ac["cvar"] / pai).map(fm.mult).values,
            "Leitura": ["⚠️ cauda mais pesada do que o ETF" if v > pai else "🟢 cauda mais leve do que o ETF"
                        for v in t_ac["cvar"]],
        })
        col_t, col_h = st.columns([3, 2])
        with col_t:
            st.dataframe(drill, hide_index=True, width="stretch")
            st.caption(f"As 4 maiores posições do {sel}. CVaR {nivel_live} do próprio ETF no mesmo período: "
                       f"{fm.pct(pai, 2)}. Uma ação com cauda mais pesada do que o ETF mostra o benefício da "
                       "diversificação dentro do setor. Descritivo: não está backtestado.")
        with col_h:
            cm = r_ac[acoes].corr()
            hm = go.Figure(go.Heatmap(z=cm.values, x=acoes, y=acoes, zmin=-1, zmax=1, colorscale="RdBu_r",
                                      text=[[fm.num(v) for v in row] for row in cm.values],
                                      texttemplate="%{text}", showscale=False))
            layout(hm, 300, "Correlação entre as ações", yaxis=dict(autorange="reversed"))
            st.plotly_chart(hm, width="stretch")
    except Exception as e:  # noqa: BLE001
        st.error(f"Erro ao processar as ações de {sel}: {e}")

    st.subheader("🗂️ Exportar para o Obsidian")
    vault = Path(st.text_input("Pasta do cofre", os.environ.get("OBSIDIAN_VAULT", str(OBSIDIAN_VAULT))))
    hoje = date.today()
    params = {"janela": janela, "confianca_cvar": alpha, "limiar_panico": panic_q,
              "dados_ate": str(rets.index[-1].date())}
    edges = [(a, b, d["corr"]) for a, b, d in sorted(grafo.edges(data=True), key=lambda e: e[2]["weight"])]
    nota = ck.obsidian_log(hoje, params, tabela.drop(columns="Tipo"), edges, regime, sel, drill)
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

# ============================================================================ 5. Fatores macro
with T_FATORES:
    if links is None:
        st.info("Ainda não há ligações medidas. Corre `python run_pipeline.py --offline`.")
    else:
        st.subheader("Como os fatores macro mexem com cada setor")
        st.caption(f"Regressão semanal de cada setor em cada fator, nos últimos {CVAR_YEARS} anos, com erros-padrão "
                   f"HAC (Newey–West). Uma ligação é aceite quando |t| ≥ {fm.num(LINK_T_MIN, 0)}. "
                   "Lê-se como: **o que acontece ao setor quando o fator se move na direção adversa** "
                   "(juros, inflação, spreads, VIX, petróleo ou dólar a subir).")
        fat = list(MACRO_FACTORS)
        nomes_f = [MACRO_FACTORS[f]["curto"] for f in fat]
        L = links.set_index(["fator", "setor"])
        # efeito do movimento adverso: + = agrava o setor (t com o sinal trocado se subir é adverso)
        efeito = pd.DataFrame({f: [-MACRO_FACTORS[f]["adverso"] * L.loc[(f, t), "t"] for t in setores] for f in fat},
                              index=setores)
        aceite = pd.DataFrame({f: [bool(L.loc[(f, t), "aceite"]) for t in setores] for f in fat}, index=setores)
        txt = [[(fm.num(v, 1) if ok else f"({fm.num(v, 1)})") for v, ok in zip(rv, ra)]
               for rv, ra in zip(efeito.values, aceite.values)]
        hm_f = go.Figure(go.Heatmap(
            z=np.clip(efeito.values, -8, 8), x=nomes_f, y=[nome(t) for t in setores], zmid=0,
            colorscale=[[0, RISCO["verde"]], [0.5, "#131c2e"], [1, RISCO["vermelho"]]], text=txt, xgap=2, ygap=2,
            texttemplate="%{text}", colorbar=dict(title="t", thickness=12),
            hovertemplate="%{y} · %{x}<br>t do efeito adverso: %{text}<extra></extra>"))
        layout(hm_f, 480, "Efeito do movimento adverso (estatística t): vermelho agrava, verde atenua",
               yaxis=dict(autorange="reversed"), xaxis=dict(side="top", tickangle=0))
        st.plotly_chart(hm_f, width="stretch")
        st.caption("Entre parênteses: não significativo (|t| < 2). A cor está limitada a |t| = 8 para não esconder "
                   "as ligações moderadas.")

        acc = links[links["aceite"].astype(bool)].copy()
        acc["efeito"] = np.where(acc["sinal"] == "+", "agrava", "atenua")
        acc = acc.reindex(acc["t"].abs().sort_values(ascending=False).index)
        st.dataframe(pd.DataFrame({
            "Fator": acc["fator"].map(lambda f: MACRO_FACTORS[f]["nome"]),
            "Setor": acc["setor"].map(nome),
            "Movimento adverso": acc["efeito"],
            "β": acc["beta"].map(lambda v: fm.num(v, 3, sinal=True)),
            "t (HAC)": acc["t"].map(lambda v: fm.num(v, 1, sinal=True)),
            "R²": acc["r2"].map(lambda v: fm.pct(v, 0)),
            "Semanas": acc["n"],
        }), hide_index=True, width="stretch")
        st.caption(f"{len(acc)} de {len(links)} ligações aceites. β: retorno semanal do setor por +1 p.p. "
                   "(juros, inflação, spreads) ou por +1% (VIX, petróleo, dólar, em log). R²: parte da variação "
                   "semanal do setor explicada pelo fator. São regressões de um fator de cada vez e, com 66 testes, "
                   "cerca de 3 ligações podem ser falsos positivos.")

# ============================================================================ 6. Modelo (backtest)
with T_MODELO:
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

# ============================================================================ 7. Carteira e stress
with T_CARTEIRA:
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

# ============================================================================ 8. Glossário
with T_GLOSSARIO:
    st.subheader("Glossário")
    st.markdown(f"""
| Termo | Significado |
|---|---|
| **VaR {N_CVAR}** | Perda a 1 dia que só deve ser ultrapassada em 2,5% dos dias (1 em 40). Em perda positiva. |
| **ES / CVaR {N_CVAR}** | Perda **média** nos dias em que o VaR é ultrapassado. É a medida de cauda do FRTB (Basileia III). |
| **CVaR histórico** | Média dos piores dias da janela passada ({CVAR_YEARS} anos), sem modelo. Muda devagar. |
| **ES previsto ({OFICIAL})** | Previsão para amanhã: volatilidade atual do GARCH × forma da cauda dos resíduos passados. |
| **Exceção** | Dia em que a perda real ultrapassou o VaR previsto na véspera. |
| **Kupiec / Christoffersen / Z2** | Testes ao modelo: frequência das exceções / se aparecem em grupos / tamanho das perdas face ao ES. |
| **Semáforo de Basileia** | Verde, amarelo ou vermelho pelas exceções dos últimos {BT_TL_DAYS} dias. Fora do verde há **risco de modelo**. |
| **Downside beta (β↓)** | Quanto o setor cai por cada 1% de queda do SPY, só nos dias de pânico (os {fm.pct(PANIC_Q, 0)} piores). |
| **Score de risco (0–100)** | Soma ponderada de cauda, sistémico, stress, regime e momento, cada um ordenado entre os 11 setores. |
| **P × I** | Probabilidade (1–5) × impacto (1–5). Níveis: {", ".join(f"{n} até {lim}" for lim, n in PI_LEVELS)}. |
| **Regime de juros** | Subida ou descida se o Treasury 10Y mudou ≥ {fm.num(RATE_THRESHOLD_PP)} p.p. em {RATE_LOOKBACK} dias úteis; senão estável. |
| **Contribuição de Euler** | Parte do risco da carteira atribuível a cada setor; as partes somam o total. |
| **Diversificação (×)** | Soma dos riscos isolados ÷ risco da carteira. 1× = nenhuma diversificação. |
| **Ledoit-Wolf** | Covariância "encolhida" para uma matriz simples: menos ruído quando há poucos dados por ativo. |
| **MST** | Árvore que liga os setores pelas correlações mais fortes: mostra por onde o contágio passa. |
| **HAC (Newey–West)** | Erros-padrão robustos a autocorrelação e volatilidade variável, para o t das ligações macro. |
| **Cíclico / sensível / defensivo** | Classificação económica do setor (não é nível de risco). |
""")
    st.caption("Metodologia completa, com fórmulas e referências: docs/nota-metodologica.md (também em PDF).")
