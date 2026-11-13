"""Estilo, constantes e funções de apresentação comuns aos separadores do cockpit (sem dados do run)."""
import pandas as pd
import plotly.graph_objects as go
from config.settings import CVAR_LEVEL, OFFICIAL_VAR_METHOD, PI_LEVELS, SECTORS
from src.report import formato as fm
from src.report.alerts import pi_min

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


METODOS = {"historico": "Histórico", "garch": "GJR-GARCH(1,1)-t", "fhs": "GARCH-FHS", "evt": "GARCH-EVT"}


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


def modelo_txt(linha) -> str:
    """Estado do modelo oficial numa empresa (linha de company_risk.csv)."""
    if linha is None:
        return "sem modelo"
    if bool(linha["validado"]):
        return "✅ validado"
    return f"⚠ ES corrigido ×{fm.num(linha['mult_es'])}" if bool(linha["es_subestimado"]) else "⚠ falha um teste"


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
