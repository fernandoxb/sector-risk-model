"""Bot Sentinela: compara o estado de hoje com o do último dia de dados e gera alertas.

Funções puras (testáveis com estados sintéticos):
  pi_level        nível do score P × I (baixo, moderado, elevado, crítico)
  build_state     estado compacto de um dia: score, semáforo, P × I, exceção ao VaR oficial, ligações, regime
  compare         lista de alertas entre o estado anterior e o atual
  render_alert    linha Markdown de um alerta (campos inline do Dataview)
  update_alerts   novo texto de Alertas.md (secção do dia substituída, mais recente no topo)
  notification    texto da notificação macOS (só severidades em ALERT_NOTIFY)

Alertas (severidade):
  semáforo        piora para vermelho = alta; piora para amarelo = média; melhoria = baixa
  exceção         perda acima do VaR 97,5% oficial = média; acima também do ES 97,5% = alta
  nível P × I     sobe para o nível mais alto = alta; outra subida = média; descida = baixa
  ligação medida  entra = média; sai = baixa
  regime de juros muda para subida = alta; outra mudança = média

I/O só em main(): lê data/features, grava data/state/estado_<data>.json, escreve
<cofre>/Risco Setorial/Alertas.md e envia a notificação (osascript).
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (ALERT_KEEP_DAYS, ALERT_NOTIFY, ALERT_SEVERITIES, CLEAN, FEATURES,  # noqa: E402
                             MACRO_FACTORS, OBSIDIAN_FOLDER, OBSIDIAN_VAULT, OFFICIAL_MEASURES,
                             OFFICIAL_VAR_METHOD, PI_LEVELS, RATE_LOOKBACK, RATE_THRESHOLD_PP,
                             SEMAFORO_ORDEM, STATE)
from src.collect.sessao import ultima_sessao  # noqa: E402
from src.models import cockpit as ck  # noqa: E402

VAR_KEY, ES_KEY = "var975", "es975"
LEVEL_975 = OFFICIAL_MEASURES[VAR_KEY][1]
NIVEL_TXT = f"{LEVEL_975:.1%}".replace(".", ",")          # "97,5%"
METODO_TXT = OFFICIAL_VAR_METHOD.upper()


def _pct(x: float) -> str:
    return f"{x:.1%}".replace(".", ",")


def _num(x: float, fmt: str = ".1f") -> str:
    return format(x, fmt).replace(".", ",")


# ----------------------------------------------------------------------------- estado do dia
def pi_level(p: int, i: int, levels=PI_LEVELS) -> str:
    """Nível de P × I: o primeiro nível cujo limite superior é >= P × I."""
    pi = int(p) * int(i)
    return next(nome for lim, nome in levels if pi <= lim)


def pi_min(nivel: str, levels=PI_LEVELS) -> int:
    """Menor P × I que pertence ao `nivel` (ex.: crítico → 17)."""
    i = [n for _, n in levels].index(nivel)
    return levels[i - 1][0] + 1 if i else 1


def build_state(asof: str, scorecard: pd.DataFrame, series: pd.DataFrame, links: pd.DataFrame,
                regime: dict | None) -> dict:
    """Estado compacto do dia `asof` (data dos dados).

    scorecard: índice = setor, colunas score_final, semaforo, impacto, probabilidade, rank
    series:    var_backtest_series (data, setor, ret, var_<método>_<nível>, es_<método>_<nível>)
    links:     factor_links.csv (fator, setor, aceite, sinal)
    """
    var_col = f"var_{OFFICIAL_VAR_METHOD}_{LEVEL_975}"
    es_col = f"es_{OFFICIAL_VAR_METHOD}_{LEVEL_975}"
    day = series[pd.to_datetime(series["data"]) == pd.Timestamp(asof)].set_index("setor")
    setores = {}
    for t, r in scorecard.iterrows():
        s = {"score": round(float(r["score_final"]), 2), "rank": int(r["rank"]), "semaforo": r["semaforo"],
             "impacto": int(r["impacto"]), "probabilidade": int(r["probabilidade"]),
             "pi": int(r["impacto"]) * int(r["probabilidade"]),
             "nivel": pi_level(r["probabilidade"], r["impacto"])}
        if t in day.index and pd.notna(day.loc[t, var_col]):
            perda, var, es = -float(day.loc[t, "ret"]), float(day.loc[t, var_col]), float(day.loc[t, es_col])
            s |= {"perda": round(perda, 6), VAR_KEY: round(var, 6), ES_KEY: round(es, 6), "excecao": perda > var}
        setores[t] = s
    acc = links[links["aceite"].astype(bool)]
    ligacoes = sorted(f"{f}|{t}|{s}" for f, t, s in zip(acc["fator"], acc["setor"], acc["sinal"]))
    return {"asof": str(asof), "setores": setores, "ligacoes": ligacoes,
            "regime_juros": regime["regime"] if regime else None}


# ----------------------------------------------------------------------------- comparação
def _alert(sev: str, tipo: str, alvo: str, texto: str) -> dict:
    return {"severidade": sev, "tipo": tipo, "alvo": alvo, "texto": texto}


def _fator_nome(f: str) -> str:
    return MACRO_FACTORS.get(f, {}).get("nome", f)


def compare(prev: dict | None, cur: dict) -> list[dict]:
    """Alertas entre o estado anterior (None = primeiro run) e o atual, ordenados por severidade."""
    out = []
    # exceções ao VaR oficial no próprio dia: não precisam do estado anterior
    for t, s in cur["setores"].items():
        if s.get("excecao"):
            acima_es = s["perda"] > s[ES_KEY]
            out.append(_alert("alta" if acima_es else "média", "exceção", t,
                              f"[[{t}]]: perda de {_pct(s['perda'])} acima do VaR {NIVEL_TXT} {METODO_TXT} "
                              f"({_pct(s[VAR_KEY])})" + (f" e do ES {NIVEL_TXT} ({_pct(s[ES_KEY])})" if acima_es else "")))
    if prev is not None:
        top = PI_LEVELS[-1][1]
        ordem_nivel = [n for _, n in PI_LEVELS]
        for t, s in cur["setores"].items():
            p = prev["setores"].get(t)
            if p is None:
                continue
            a, b = SEMAFORO_ORDEM.index(p["semaforo"]), SEMAFORO_ORDEM.index(s["semaforo"])
            if a != b:
                sev = "baixa" if b < a else "alta" if s["semaforo"] == SEMAFORO_ORDEM[-1] else "média"
                out.append(_alert(sev, "semáforo", t, f"[[{t}]]: semáforo do VaR {METODO_TXT} "
                                                      f"{p['semaforo']} → {s['semaforo']}"))
            a, b = ordem_nivel.index(p["nivel"]), ordem_nivel.index(s["nivel"])
            if a != b:
                sev = "baixa" if b < a else "alta" if s["nivel"] == top else "média"
                out.append(_alert(sev, "nível P×I", t,
                                  f"[[{t}]]: nível {p['nivel']} → {s['nivel']} (P×I {p['pi']} → {s['pi']}, "
                                  f"score {_num(p['score'])} → {_num(s['score'])})"))
        antes, agora = set(prev["ligacoes"]), set(cur["ligacoes"])
        for sev, verbo, ligs in (("média", "nova", agora - antes), ("baixa", "deixou de ser significativa", antes - agora)):
            for lig in sorted(ligs):
                f, t, sinal = lig.split("|")
                efeito = "agrava" if sinal == "+" else "atenua"
                out.append(_alert(sev, "ligação", f"{f}→{t}",
                                  f"Ligação medida {verbo}: {_fator_nome(f)} → [[{t}]] (o adverso {efeito})"))
        r0, r1 = prev.get("regime_juros"), cur.get("regime_juros")
        if r0 and r1 and r0 != r1:
            out.append(_alert("alta" if r1 == "subida" else "média", "regime de juros", "juros",
                              f"Regime de juros: {r0} → {r1} (Treasury 10Y, {RATE_LOOKBACK} dias úteis)"))
    return sorted(out, key=lambda x: ALERT_SEVERITIES.index(x["severidade"]))


def stale_alert(asof: str, esperado: str) -> dict | None:
    """Alerta de severidade alta se os dados acabam antes da última sessão de NY já fechada."""
    if asof >= esperado:
        return None
    return _alert("alta", "dados", "pipeline", f"Dados atrasados: os preços acabam em {asof}, mas a última sessão "
                                               f"fechada em Nova Iorque é {esperado} (falha na coleta ou feriado).")


# ----------------------------------------------------------------------------- Obsidian
def render_alert(asof: str, a: dict) -> str:
    return f"- [data:: {asof}] [severidade:: {a['severidade']}] [tipo:: {a['tipo']}] {a['texto']}"


def _sections(text: str) -> dict[str, str]:
    """Secções '## AAAA-MM-DD' de um Alertas.md existente → {data: corpo}."""
    secs, cur = {}, None
    for line in text.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            secs[cur] = []
        elif cur is not None:
            secs[cur].append(line)
    return {k: "\n".join(v).strip() for k, v in secs.items()}


def update_alerts(existing: str | None, asof: str, alerts: list[dict], first_run: bool = False,
                  keep: int = ALERT_KEEP_DAYS) -> str:
    """Texto completo de Alertas.md com a secção de `asof` (re)escrita; mais recente no topo."""
    secs = _sections(existing or "")
    if alerts:
        body = "\n".join(render_alert(asof, a) for a in alerts)
    else:
        body = "- Sem alertas."
    if first_run:
        body += "\n- Primeiro estado gravado: sem dia anterior para comparar semáforos, níveis, ligações e regime."
    secs[asof] = body
    days = sorted(secs, reverse=True)[:keep]
    n = {s: sum(a["severidade"] == s for a in alerts) for s in ALERT_SEVERITIES}
    fm = ["---", "tipo: alertas", f"ultimo_dia: {asof}", f"alertas_ultimo_dia: {len(alerts)}",
          *[f"{k.replace('é', 'e')}_ultimo_dia: {v}" for k, v in n.items()], "---"]
    head = ["# Alertas de risco setorial", "",
            "Gerados pelo bot Sentinela (`src/report/alerts.py`) em cada run do pipeline. Mais recente no topo.", ""]
    out = fm + [""] + head
    for d in days:
        out += [f"## {d}", "", secs[d], ""]
    return "\n".join(out)


def notification(alerts: list[dict], notify=ALERT_NOTIFY) -> str | None:
    """Texto da notificação (None se não houver alertas das severidades a notificar)."""
    hi = [a for a in alerts if a["severidade"] in notify]
    if not hi:
        return None
    txt = "; ".join(a["texto"].replace("[[", "").replace("]]", "") for a in hi[:3])
    return txt + (f" (+{len(hi) - 3})" if len(hi) > 3 else "")


def send_notification(text: str) -> None:
    """Notificação macOS via osascript (não faz nada fora do macOS)."""
    if sys.platform != "darwin":
        return
    esc = text.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run(["osascript", "-e", f'display notification "{esc}" with title "Risco setorial" '
                                       f'subtitle "Alerta de severidade alta"'], check=False)


def score_history(states: list[dict]) -> pd.DataFrame:
    """Uma linha por (data, setor) a partir dos estados gravados: score, posição, P × I, nível e semáforo."""
    rows = [{"data": st["asof"], "setor": t, **{k: v[k] for k in ("score", "rank", "pi", "nivel", "semaforo")}}
            for st in states for t, v in st["setores"].items()]
    cols = ["data", "setor", "score", "rank", "pi", "nivel", "semaforo"]
    return pd.DataFrame(rows, columns=cols).sort_values(["data", "setor"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- estados gravados
def state_path(asof: str) -> Path:
    return STATE / f"estado_{asof}.json"


def previous_state(asof: str, folder: Path = STATE) -> dict | None:
    """Último estado gravado com data de dados anterior a `asof`."""
    olds = sorted(p for p in folder.glob("estado_*.json") if p.stem.removeprefix("estado_") < asof)
    return json.loads(olds[-1].read_text()) if olds else None


def load_current() -> tuple[dict, pd.DataFrame]:
    """Estado de hoje a partir de data/features (+ séries do backtest para o relatório semanal)."""
    official = pd.read_csv(FEATURES / "official_risk.csv")
    asof = str(official["dados_ate"].iloc[0])
    sc = pd.read_csv(FEATURES / "scorecard.csv", index_col="setor")
    series = pd.read_parquet(FEATURES / "var_backtest_series.parquet")
    links = pd.read_csv(FEATURES / "factor_links.csv")
    regime = None
    fac = CLEAN / "factors.parquet"
    if fac.exists():
        regime = ck.rate_regime(pd.read_parquet(fac)["juros"], RATE_LOOKBACK, RATE_THRESHOLD_PP)
    return build_state(asof, sc, series, links, regime), series


def main(notify: bool = True) -> list[dict]:
    cur, _ = load_current()
    prev = previous_state(cur["asof"])
    alerts = compare(prev, cur)
    atraso = stale_alert(cur["asof"], ultima_sessao().isoformat())
    if atraso:
        alerts.insert(0, atraso)

    STATE.mkdir(parents=True, exist_ok=True)
    state_path(cur["asof"]).write_text(json.dumps(cur, ensure_ascii=False, indent=1))

    folder = OBSIDIAN_VAULT / OBSIDIAN_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    note = folder / "Alertas.md"
    note.write_text(update_alerts(note.read_text() if note.exists() else None, cur["asof"], alerts,
                                  first_run=prev is None))
    # para o cockpit e o snapshot: histórico do score (todos os estados) e alertas do último run
    states = [json.loads(p.read_text()) for p in sorted(STATE.glob("estado_*.json"))]
    score_history(states).to_csv(FEATURES / "score_history.csv", index=False, float_format="%.2f")
    # alertas do último run
    (FEATURES / "alerts.json").write_text(json.dumps(
        {"asof": cur["asof"], "referencia": prev["asof"] if prev else None, "alertas": alerts},
        ensure_ascii=False, indent=1))
    msg = notification(alerts)
    if notify and msg:
        send_notification(msg)
    ref = prev["asof"] if prev else "nenhum"
    n = {s: sum(a["severidade"] == s for a in alerts) for s in ALERT_SEVERITIES}
    print(f"[sentinela] {cur['asof']} vs {ref}: {len(alerts)} alertas {n}; {note}")
    return alerts


if __name__ == "__main__":
    main(notify="--sem-notificacao" not in sys.argv)
