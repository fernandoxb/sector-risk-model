"""Cockpit de Risco Setorial Interativo.

Correr a partir da raiz do projeto:
    .venv/bin/streamlit run app/cockpit/app.py
Sem rede, ou com o Yahoo a limitar pedidos, usa o snapshot versionado em app/cockpit/snapshot/
(COCKPIT_SNAPSHOT=1 força esse modo, como no Streamlit Community Cloud).
Separadores: Resumo · Mapa de risco (setor → ficha do setor e empresas → ficha da empresa) · Minha carteira ·
Setores · Análise ao vivo · Fatores macro · Modelo · Carteira-base · Glossário. Tema em .streamlit/config.toml.
Metodologia: docs/nota-metodologica.md
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(AQUI)]          # config/ e src/ (raiz) e paineis/ (esta pasta)

# O Streamlit volta a correr este ficheiro a cada interação, mas não recarrega os módulos já importados. Depois de
# uma atualização do código (ex.: novo deploy no Streamlit Cloud), este ficheiro seria novo e config/, src/ e
# paineis/ antigos, o que dá ImportError. Por isso, se algum ficheiro do projeto mudou desde o último run,
# esquecem-se esses módulos.
_assinatura = max(f.stat().st_mtime for d in (ROOT / "config", ROOT / "src", AQUI / "paineis") for f in d.rglob("*.py"))
if getattr(sys, "_cockpit_assinatura", None) != _assinatura:
    for _m in [m for m in sys.modules if m.split(".")[0] in ("config", "src", "paineis")]:
        del sys.modules[_m]
    sys._cockpit_assinatura = _assinatura
from config.settings import (CALENDAR_PREVIEW, COMPANY_TTL, CVAR_YEARS, FEATURES, MARKET, RATE_LOOKBACK, RATE_SERIES,
                             RATE_THRESHOLD_PP, SECTORS, SNAPSHOT)  # noqa: E402
from src.collect import empresas as emp, sessao as cal  # noqa: E402
from src.models import backtest_empresas as be, cockpit as ck  # noqa: E402
import paineis.ao_vivo  # noqa: E402
import paineis.carteira_base  # noqa: E402
import paineis.fatores  # noqa: E402
import paineis.glossario  # noqa: E402
import paineis.mapa  # noqa: E402
import paineis.minha_carteira  # noqa: E402
import paineis.modelo  # noqa: E402
import paineis.resumo  # noqa: E402
import paineis.setores  # noqa: E402
from paineis.comum import OFICIAL  # noqa: E402

st.set_page_config(page_title="Cockpit de Risco Setorial", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")


def es_emp(t: str):
    """ES 97,5% previsto de uma empresa, já com a correção do backtest (None se não tiver modelo)."""
    if comp_risk is None or t not in comp_risk.index:
        return None
    return float(comp_risk.loc[t, "es975_aj"])


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
bt_split = feat_csv("var_backtest_split.csv")
bt_multi = feat_csv("var_backtest_multiple.csv")
comp_bt = feat_csv("company_backtest.csv")
comp_risk = feat_csv("company_risk.csv", index_col="empresa")
hold = feat_csv("holdings.csv")
if hold is None:
    hold = emp.fallback_holdings()
SNAP_EMP = json.loads((SNAPSHOT / "companies.json").read_text()) if (SNAPSHOT / "companies.json").exists() else {}


@st.cache_data(ttl=COMPANY_TTL, show_spinner="📰 A ler a ficha e as notícias da empresa…")
def empresa_vivo(ticker: str) -> dict:
    return {"info": emp.fetch_info(ticker), "noticias": emp.fetch_news(ticker)}


@st.cache_data(ttl=3600, show_spinner="📈 A testar o modelo na tua carteira…")
def backtest_carteira(chave: str, _rets: pd.DataFrame, _w: pd.Series, ate: str) -> dict | None:
    """Backtest e previsão FHS da carteira; `chave` (pesos) e `ate` (último preço) identificam o resultado em cache."""
    return be.portfolio_backtest(_rets, _w)


def empresa_dados(ticker: str) -> tuple[dict, str]:
    """Ficha e notícias: ao vivo (Yahoo) ou, se falhar, as do snapshot. Devolve também a origem."""
    if not (FORCAR_SNAPSHOT or st.session_state.get("yahoo_falhou")):
        try:
            return empresa_vivo(ticker), "ao vivo"
        except Exception:  # noqa: BLE001
            pass
    return SNAP_EMP.get(ticker, {"info": {}, "noticias": []}), f"snapshot de {SNAP_META.get('dados_ate', '?')}"
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
                   "para a sessão seguinte ao último fecho.")
hoje_ny = pd.Timestamp.now(tz="America/New_York").date()
ult = cal.ultima_sessao()
prox = cal.proximo_dia_util(ult)
ant = cal.proximo_fecho_antecipado(hoje_ny)
DIAS = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]


def dia_txt(d) -> str:
    return f"{DIAS[d.weekday()]} {d.day:02d}/{d.month:02d}"


st.sidebar.markdown("**Calendário da bolsa (NYSE)**")
st.sidebar.markdown(
    f"- Última sessão fechada: **{dia_txt(ult)}**\n"
    f"- Próxima sessão: **{dia_txt(prox)}**" + (" (fecha às 13:00)" if cal.fecho_antecipado(prox) else "") + "\n"
    + "\n".join(f"- Fechada: **{dia_txt(f)}** · {n}" for f, n in cal.proximos_feriados(hoje_ny, CALENDAR_PREVIEW))
    + f"\n- Fecho às 13:00: **{dia_txt(ant)}**")
if PIPE_ASOF != "—" and PIPE_ASOF < ult.isoformat():
    st.sidebar.warning(f"Os resultados do pipeline ({PIPE_ASOF}) estão atrás da última sessão fechada ({ult}).")
st.sidebar.caption("Feriados calculados pelas regras da NYSE e validados com as sessões do SPY desde 2005. "
                   "Horas de Nova Iorque; uma sessão conta como fechada 1 h depois do fecho.")
st.sidebar.markdown("---")
st.sidebar.markdown("[Código no GitHub](https://github.com/fernandoxb/sector-risk-model) · "
                    "[Nota metodológica (PDF)](https://github.com/fernandoxb/sector-risk-model/blob/main/docs/"
                    "nota-metodologica.pdf)")

# ============================================================================ cabeçalho
st.title("🛡️ Cockpit de Risco Setorial")
if "snapshot" in FONTES:
    st.info(f"📦 Modo snapshot: preços e juros guardados no repositório (até {SNAP_META.get('dados_ate', '?')}), "
            f"porque {'COCKPIT_SNAPSHOT está ativo' if FORCAR_SNAPSHOT else 'o Yahoo não respondeu'}.")

tabs = st.tabs(["📋 Resumo", "🕸️ Mapa de risco", "💼 Minha carteira", "🎯 Setores", "📊 Análise ao vivo", "🔗 Fatores macro",
                "🧪 Modelo", "🧩 Carteira-base", "📖 Glossário"])
T_RESUMO, T_MAPA, T_MINHA, T_SETORES, T_VIVO, T_FATORES, T_MODELO, T_CARTEIRA, T_GLOSSARIO = tabs

# Cada separador está em paineis/<nome>.py e recebe os dados deste run (as variáveis deste ficheiro).
G = globals()
for _tab, _painel in ((T_RESUMO, paineis.resumo), (T_MAPA, paineis.mapa), (T_MINHA, paineis.minha_carteira),
                      (T_SETORES, paineis.setores), (T_VIVO, paineis.ao_vivo), (T_FATORES, paineis.fatores),
                      (T_MODELO, paineis.modelo), (T_CARTEIRA, paineis.carteira_base),
                      (T_GLOSSARIO, paineis.glossario)):
    with _tab:
        _painel.render(G)
