"""Relatório semanal no Obsidian: Risco Setorial/Semana AAAA-Www.md (+ 00 Painel.md).

Sai na última sessão da NYSE de cada semana (normalmente sexta; quinta se a sexta for feriado, como a
Sexta-feira Santa) ou com --semanal. Conteúdo:
  - top 3 do scorecard e variação face ao último estado da semana anterior (score e posição);
  - exceções ao VaR 97,5% oficial na semana (perda > VaR previsto, só com dados até t-1);
  - ligações medidas que entraram ou saíram face à semana anterior;
  - cenário de stress mais severo para a carteira e o setor mais atingido nesse cenário.
Frontmatter estável para o Dataview (tipo: relatorio-semanal).

Funções puras: week_id, week_start, week_exceptions, worst_stress, weekly_note, painel_note.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (FEATURES, OBSIDIAN_FOLDER, OBSIDIAN_VAULT, OFFICIAL_MEASURES,  # noqa: E402
                             OFFICIAL_VAR_METHOD, SECTORS, STATE, WEEKLY_TOP)
from src.collect.sessao import ultima_sessao_da_semana  # noqa: E402
from src.report.alerts import METODO_TXT, NIVEL_TXT, _fator_nome, _num, _pct, load_current, previous_state  # noqa: E402

LEVEL_975 = OFFICIAL_MEASURES["var975"][1]
PORTFOLIO = "Carteira"            # nome da série da carteira em stress_tests.csv


def week_id(day) -> str:
    y, w, _ = pd.Timestamp(day).isocalendar()
    return f"{y}-W{w:02d}"


def week_start(day) -> pd.Timestamp:
    d = pd.Timestamp(day).normalize()
    return d - pd.Timedelta(days=d.weekday())


def is_report_day(day) -> bool:
    """Última sessão da semana ISO: a sessão seguinte já cai noutra semana."""
    return ultima_sessao_da_semana(pd.Timestamp(day).date())


def reference_state(asof: str, folder: Path = STATE) -> dict | None:
    """Estado de comparação: o último antes de segunda-feira; se não houver (primeira semana com estados),
    o primeiro estado desta semana anterior a `asof`."""
    ref = previous_state(str(week_start(asof).date()), folder)
    if ref is None:
        firsts = sorted(p for p in folder.glob("estado_*.json")
                        if str(week_start(asof).date()) <= p.stem.removeprefix("estado_") < asof)
        ref = json.loads(firsts[0].read_text()) if firsts else None
    return ref


def week_exceptions(series: pd.DataFrame, asof) -> pd.DataFrame:
    """Exceções ao VaR 97,5% oficial entre segunda e `asof` (perda positiva)."""
    var_col, es_col = f"var_{OFFICIAL_VAR_METHOD}_{LEVEL_975}", f"es_{OFFICIAL_VAR_METHOD}_{LEVEL_975}"
    d = pd.to_datetime(series["data"])
    w = series[(d >= week_start(asof)) & (d <= pd.Timestamp(asof))].copy()
    w["perda"] = -w["ret"]
    hit = w[w["perda"] > w[var_col]]
    return (hit.assign(data=pd.to_datetime(hit["data"]).dt.date, var=hit[var_col], es=hit[es_col],
                       acima_es=hit["perda"] > hit[es_col])
            [["data", "setor", "perda", "var", "es", "acima_es"]].sort_values(["data", "setor"]).reset_index(drop=True))


def worst_stress(stress: pd.DataFrame) -> dict:
    """Cenário com a maior perda da carteira e o setor mais atingido nesse cenário."""
    pf = stress[stress["serie"] == PORTFOLIO]
    row = pf.loc[pf["perda"].idxmax()]
    sec = stress[(stress["cenario"] == row["cenario"]) & stress["serie"].isin(SECTORS)]
    top = sec.loc[sec["perda"].idxmax()]
    return {"cenario": row["cenario"], "perda_carteira": float(row["perda"]), "proxy": bool(row["proxy"]),
            "setor": top["serie"], "perda_setor": float(top["perda"]), "proxy_setor": bool(top["proxy"])}


def _yaml_list(xs) -> str:
    return "[" + ", ".join(xs) + "]"


def week_days(series: pd.DataFrame, asof) -> int:
    """Dias com dados entre segunda e `asof`."""
    d = pd.to_datetime(series["data"])
    return int(d[(d >= week_start(asof)) & (d <= pd.Timestamp(asof))].nunique())


def weekly_note(cur: dict, ref: dict | None, exc: pd.DataFrame, stress: dict, dias: int) -> tuple[dict, str]:
    """(frontmatter, Markdown) do relatório da semana de cur['asof']; ref = último estado da semana anterior;
    dias = dias com dados na semana (para as exceções esperadas)."""
    asof = cur["asof"]
    rank = sorted(cur["setores"].items(), key=lambda kv: kv[1]["rank"])
    top = rank[:WEEKLY_TOP]
    novas = sorted(set(cur["ligacoes"]) - set(ref["ligacoes"])) if ref else []
    sairam = sorted(set(ref["ligacoes"]) - set(cur["ligacoes"])) if ref else []
    fm = {"tipo": "relatorio-semanal", "semana": week_id(asof), "inicio": str(week_start(asof).date()),
          "dados_ate": asof, "referencia": ref["asof"] if ref else "nenhuma",
          "top1": top[0][0], "top1_score": round(top[0][1]["score"], 1),
          "top3": _yaml_list(t for t, _ in top), "excecoes": len(exc), "excecoes_acima_es": int(exc["acima_es"].sum()),
          "ligacoes_novas": len(novas), "ligacoes_sairam": len(sairam),
          "stress_cenario": f"\"{stress['cenario']}\"", "stress_perda_carteira": round(stress["perda_carteira"], 4),
          "regime_juros": cur.get("regime_juros") or "—"}

    L = ["---", *[f"{k}: {v}" for k, v in fm.items()], "---", "",
         f"# Risco setorial — semana {fm['semana']}", "",
         f"Dados de {fm['inicio']} a {asof}. Variações face ao estado de "
         f"{ref['asof'] if ref else '— (sem estados anteriores gravados)'}"
         + (" (início desta semana: ainda não há estado da semana anterior)."
            if ref and ref["asof"] >= fm["inicio"] else "."), "",
         f"## Top {WEEKLY_TOP} do scorecard (0–100)", "",
         "| # | Setor | Score | Variação | P × I | Nível | Semáforo |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for t, s in top:
        if ref and t in ref["setores"]:
            r = ref["setores"][t]
            mov = r["rank"] - s["rank"]
            d = s["score"] - r["score"]
            var = ("=" if abs(d) < 0.05 else _num(d, "+.1f")) + (f" ({mov:+d} pos.)" if mov else "")
        else:
            var = "—"
        L.append(f"| {s['rank']} | [[{t}]] {SECTORS[t]} | {_num(s['score'])} | {var} | "
                 f"{s['probabilidade']} × {s['impacto']} = {s['pi']} | {s['nivel']} | {s['semaforo']} |")

    L += ["", f"## Exceções ao VaR {NIVEL_TXT} oficial ({METODO_TXT}, 1 dia)", ""]
    if exc.empty:
        L.append("Nenhuma exceção esta semana.")
    else:
        L += ["| Data | Setor | Perda | VaR previsto | ES previsto | Acima do ES |", "| --- | --- | --- | --- | --- | --- |"]
        L += [f"| {r.data} | [[{r.setor}]] | {_pct(r.perda)} | {_pct(r.var)} | {_pct(r.es)} | {'sim' if r.acima_es else 'não'} |"
              for r in exc.itertuples()]
    esperadas = (1 - LEVEL_975) * dias * len(SECTORS)
    L.append(f"\nEsperadas com o VaR {NIVEL_TXT}: {_num(esperadas)} ({_pct(1 - LEVEL_975)} × {dias} dias × "
             f"{len(SECTORS)} setores); observadas: {len(exc)}.")

    L += ["", "## Ligações medidas fator → setor", ""]
    if ref is None:
        L.append("Sem estado anterior para comparar.")
    elif not novas and not sairam:
        L.append(f"Sem alterações face a {ref['asof']}.")
    for lig, verbo in [(x, "entrou") for x in novas] + [(x, "saiu") for x in sairam]:
        f, t, sinal = lig.split("|")
        L.append(f"- {verbo}: {_fator_nome(f)} → [[{t}]] ({'agrava' if sinal == '+' else 'atenua'})")

    L += ["", "## Stress mais severo para a carteira", "",
          f"**{stress['cenario']}**: perda da carteira de {_pct(stress['perda_carteira'])}"
          + (" (com proxy β × SPY nalguns setores)" if stress["proxy"] else "")
          + f"; setor mais atingido [[{stress['setor']}]] ({_pct(stress['perda_setor'])}"
          + (", proxy" if stress["proxy_setor"] else "") + ").", ""]
    return fm, "\n".join(L)


def painel_note() -> str:
    """00 Painel.md: logs diários do cockpit, relatórios semanais e alertas recentes (Dataview)."""
    folder = OBSIDIAN_FOLDER
    return "\n".join([
        "---", "tipo: painel", "---", "# Painel de Risco Setorial", "",
        "Requer o plugin comunitário **Dataview**.", "",
        "## Relatórios semanais", "",
        "```dataview",
        'TABLE dados_ate AS "Dados até", top3 AS "Top 3", top1_score AS "Score n.º 1", excecoes AS "Exceções VaR", '
        'ligacoes_novas AS "Ligações novas", stress_cenario AS "Stress mais severo", regime_juros AS "Regime de juros"',
        f'FROM "{folder}"', 'WHERE tipo = "relatorio-semanal"', "SORT semana DESC", "```", "",
        "## Alertas recentes", "",
        "```dataview",
        'TABLE WITHOUT ID L.data AS "Data", L.severidade AS "Severidade", L.tipo AS "Tipo", L.text AS "Alerta"',
        f'FROM "{folder}/Alertas"', "FLATTEN file.lists AS L", "WHERE L.severidade",
        "SORT L.data DESC", "LIMIT 25", "```", "",
        "## Logs diários do cockpit", "",
        "```dataview",
        'TABLE dados_ate AS "Dados até", janela AS "Janela", confianca_cvar AS "Confiança CVaR", '
        'dgs10 AS "Treasury 10Y", regime_juros AS "Regime de juros"',
        f'FROM "{folder}"', 'WHERE tipo = "log-risco-setorial"', "SORT data DESC", "```", ""])


def main(force: bool = False) -> Path | None:
    cur, series = load_current()
    folder = OBSIDIAN_VAULT / OBSIDIAN_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "00 Painel.md").write_text(painel_note())
    if not (force or is_report_day(cur["asof"])):
        print(f"[relatório] {cur['asof']} não é sexta: sem relatório semanal (usar --semanal para forçar)")
        return None
    ref = reference_state(cur["asof"])
    stress = worst_stress(pd.read_csv(FEATURES / "stress_tests.csv"))
    fm, md = weekly_note(cur, ref, week_exceptions(series, cur["asof"]), stress, week_days(series, cur["asof"]))
    out = folder / f"Semana {fm['semana']}.md"
    out.write_text(md)
    print(f"[relatório] {out}")
    return out


if __name__ == "__main__":
    main(force="--semanal" in sys.argv)
