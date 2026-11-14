"""Teia de risco em ECharts (protótipo da Fase 2): fatores macro → setores → empresas numa só vista.

Anéis concêntricos com posições fixas: os 6 fatores no centro, os 11 setores no anel do meio (vizinhos na árvore
de quedas conjuntas ficam lado a lado) e as empresas no anel de fora, junto ao seu setor. Ligações:
  fator → setor     as que resistem na regressão conjunta (Fase C); vermelho cheio agrava, verde tracejado atenua
  setor ↔ setor     árvore das quedas conjuntas na janela escolhida (cauda.graph)
  setor → empresa   pertença ao ETF; entre as empresas do setor aberto, a árvore das suas quedas conjuntas
Um clique num fator mostra a sua cadeia, com o choque a correr do fator para os setores e daí para as empresas.

`opcoes` é pura (devolve o dicionário do ECharts); `desenhar` é a parte Streamlit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import MACRO_FACTORS, SECTORS
from src.models import cauda as cd
from src.report import formato as fm
from paineis.comum import MUTED, RISCO, TXT, nome

R_FAT, R_SET, R_EMP, R_ABERTO = 0.30, 0.66, 0.97, 1.10
ALTURA = 760                                # px; a grelha é quadrada para os anéis ficarem redondos
LIM = 1.45                                   # meio lado do quadrado visível
COR_FATOR, COR_CAUDA, COR_FIO = "#818cf8", "#a78bfa", "#334155"
APAGADO = 0.07
JS_DICA = "function(p){return p.data.dica || '';}"
JS_CLIQUE = "function(p){ if (p.dataType === 'node') { return p.data.id + '|' + Date.now(); } }"


def _cor(x: float) -> str:
    """0 → verde, 0,5 → amarelo, 1 → vermelho (as cores de risco do cockpit)."""
    pts = [(0.0, RISCO["verde"]), (0.5, RISCO["amarelo"]), (1.0, RISCO["vermelho"])]
    x = float(np.clip(x, 0, 1))
    (x0, c0), (x1, c1) = next((pts[i], pts[i + 1]) for i in range(2) if x <= pts[i + 1][0])
    f = (x - x0) / (x1 - x0)
    a, b = (np.array([int(c[k:k + 2], 16) for k in (1, 3, 5)]) for c in (c0, c1))
    return "#" + "".join(f"{int(round(v)):02x}" for v in a + (b - a) * f)


def _polar(r: float, ang: float) -> list[float]:
    return [round(r * np.cos(ang), 4), round(r * np.sin(ang), 4)]


def ordem_setores(pares_setores: pd.DataFrame, por: str = "p_conj") -> list[str]:
    """Setores pela árvore das ligações (percurso em profundidade): os mais ligados ficam vizinhos no anel."""
    import networkx as nx
    g = cd.graph(pares_setores, by=por)
    if g.number_of_nodes() == 0:
        return list(SECTORS)
    ordem = list(nx.dfs_preorder_nodes(g, source=max(g.degree, key=lambda x: x[1])[0]))
    return ordem + [s for s in SECTORS if s not in ordem]


def opcoes(sc: pd.DataFrame, links: pd.DataFrame | None, pares: pd.DataFrame | None, hold: pd.DataFrame,
           comp_risk: pd.DataFrame | None, comp_fat: pd.DataFrame | None, aberto: str | None = None,
           foco: str | None = None, q_txt: str = "5%", cenario: pd.Series | None = None, por: str = "p_conj",
           carteira: set[str] | None = None, diretos: dict | None = None) -> dict:
    """Dicionário do ECharts. `pares`: quedas conjuntas da janela (todos os grupos); `aberto`: setor com as empresas
    em destaque; `foco`: fator ou setor cuja cadeia se anima; `cenario`: retorno mediano de um choque por setor e
    empresa (ticker), que passa a dar a cor (vermelho perde, verde ganha) e entra nas dicas; `por`: ligar os setores
    pelas quedas conjuntas ("p_conj") ou pela correlação ("corr"); `carteira`: tickers a destacar (contorno branco);
    `diretos`: {(fator, setor): retorno} do canal direto do fator no cenário, escrito na ligação."""
    diretos = diretos or {}
    carteira = carteira or set()
    # escala da cor = o maior efeito entre os setores (as empresas mais extremas ficam no tom máximo)
    esc = max(float(cenario.reindex(list(SECTORS)).abs().max()), 0.005) if cenario is not None and len(cenario) else None

    def cor_cen(t: str, pad: str) -> str:
        """Cor pelo cenário (divergente, com cinzento no zero) ou a cor de risco habitual."""
        if esc is None or t not in cenario.index or cenario[t] != cenario[t]:
            return pad
        return _cor(0.5 - 0.5 * float(np.clip(cenario[t] / esc, -1, 1))) if abs(cenario[t]) > 0.1 * esc else MUTED

    def txt_cen(t: str) -> str:
        return f"<br/><b>No cenário: {fm.pct(cenario[t], 1, sinal=True)}</b> (mediana)" if esc and t in cenario.index else ""
    p_set = pares[pares["grupo"] == cd.GRUPO_SETORES] if pares is not None else pd.DataFrame(columns=["a", "b", "p_conj"])
    ordem = ordem_setores(p_set, por) if len(p_set) else list(SECTORS)
    ang_s = {s: np.pi / 2 - 2 * np.pi * k / len(ordem) for k, s in enumerate(ordem)}
    fatias = 2 * np.pi / len(ordem)

    # --- ligações fator → setor (regressão conjunta; sem ela, as de um fator de cada vez)
    lf = pd.DataFrame(columns=["fator", "setor", "beta", "t", "sinal"])
    if links is not None:
        multi = "aceite_multi" in links
        sel = links[links["aceite_multi" if multi else "aceite"].astype(bool)]
        lf = pd.DataFrame({"fator": sel["fator"], "setor": sel["setor"],
                           "beta": sel["beta_pad_multi" if multi else "beta_pad"],
                           "t": sel["t_multi" if multi else "t"], "sinal": sel["sinal_multi" if multi else "sinal"]})

    # --- fatores no centro, cada um virado para os setores em que mexe
    alvo = {}
    for f in MACRO_FACTORS:
        d = lf[lf["fator"] == f]
        if len(d):
            w = d["beta"].abs().to_numpy()
            a = np.array([ang_s[s] for s in d["setor"]])
            alvo[f] = float(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum()))
        else:
            alvo[f] = 0.0
    por_ang = sorted(MACRO_FACTORS, key=lambda f: alvo[f])
    rot = np.mean([alvo[f] - 2 * np.pi * k / len(por_ang) for k, f in enumerate(por_ang)])
    ang_f = {f: rot + 2 * np.pi * k / len(por_ang) for k, f in enumerate(por_ang)}

    # --- cadeia em foco: o que fica aceso
    aceso: set[str] | None = None
    if foco in MACRO_FACTORS:
        ss = set(lf.loc[lf["fator"] == foco, "setor"])
        aceso = {foco} | ss | {f"{s}/{t}" for s, t in zip(hold["setor"], hold["ticker"]) if s in ss}
    elif foco in SECTORS:
        viz = set(p_set.loc[p_set["a"] == foco, "b"]) | set(p_set.loc[p_set["b"] == foco, "a"])
        aceso = {foco} | set(lf.loc[lf["setor"] == foco, "fator"]) | {f"{foco}/{t}" for t in hold.loc[hold["setor"] == foco, "ticker"]}
        arv = cd.graph(p_set, by=por) if len(p_set) else None
        aceso |= set(arv.neighbors(foco)) if arv is not None and foco in arv else viz

    def opac(i: str, base: float = 1.0) -> float:
        return base if aceso is None or i in aceso else APAGADO

    nos, arestas = [], []
    # fatores
    for f, a in ang_f.items():
        d = lf[lf["fator"] == f]
        ag = ", ".join(d.loc[d["sinal"] == "+", "setor"]) or "—"
        at = ", ".join(d.loc[d["sinal"] != "+", "setor"]) or "—"
        nos.append({"id": f, "name": MACRO_FACTORS[f]["curto"], "value": _polar(R_FAT, a), "category": 0,
                    "symbol": "roundRect", "symbolSize": [92, 24],
                    "itemStyle": {"color": "#1e1b4b", "borderColor": COR_FATOR, "borderWidth": 1.5, "opacity": opac(f)},
                    "label": {"show": True, "color": TXT, "fontSize": 11, "opacity": opac(f)},
                    "dica": f"<b>{MACRO_FACTORS[f]['nome']}</b><br/>A subir agrava: {ag}<br/>A subir atenua: {at}"
                            "<br/><i>clica para ver a cadeia</i>"})
    # setores
    es_s = sc["bruto_cauda"] if "bruto_cauda" in sc else pd.Series(np.nan, index=sc.index)
    es_n = (es_s - es_s.min()) / (es_s.max() - es_s.min() or 1)
    for s, a in ang_s.items():
        sco = float(sc.loc[s, "score_final"]) if s in sc.index else 50.0
        vz = p_set[(p_set["a"] == s) | (p_set["b"] == s)].nlargest(3, "p_conj")
        cai = "; ".join(f"{r.b if r.a == s else r.a} {fm.pct(r.p_conj, 0)}" for r in vz.itertuples())
        nos.append({"id": s, "name": s, "value": _polar(R_SET, a), "category": 1,
                    "symbolSize": float(36 + 22 * es_n.get(s, 0.5)),
                    "itemStyle": {"color": cor_cen(s, _cor(sco / 100)),
                                  "borderColor": TXT if s in carteira else "#0b1220", "borderWidth": 3 if s in carteira else 2,
                                  "shadowBlur": 18 if s in (aberto, foco) else 0, "shadowColor": TXT,
                                  "opacity": opac(s)},
                    "label": {"show": True, "color": "#0b1220", "fontWeight": "bold", "fontSize": 10 if len(s) > 3 else 11,
                              "opacity": opac(s),
                              **({"formatter": f"{s}\n{fm.pct(cenario[s], 1, sinal=True)}", "lineHeight": 13}
                                 if esc and s in cenario.index else {})},
                    "dica": f"<b>{nome(s)}</b><br/>Score de risco {fm.num(sco, 0)}/100"
                            + (f" · grupo {int(sc.loc[s, 'grupo'])} de {int(sc['grupo'].max())}" if "grupo" in sc and s in sc.index else "")
                            + f"<br/>ES 97,5% amanhã {fm.pct(es_s.get(s))}<br/>Cai junto com: {cai or '—'}"
                            + txt_cen(s) + "<br/><i>clica para abrir as empresas</i>"})
    # empresas
    es_e = comp_risk["es975_aj"] if comp_risk is not None else pd.Series(dtype=float)
    pct_e = es_e.rank(pct=True)
    for s in ordem:
        h = hold[hold["setor"] == s].sort_values("peso", ascending=False).reset_index(drop=True)
        n = len(h)
        abre = s == aberto
        leque = fatias * (2.4 if abre else 0.80)    # o setor aberto abre as empresas num leque mais largo
        for j, r in h.iterrows():
            a = ang_s[s] + leque * ((j + 0.5) / n - 0.5)
            i = f"{s}/{r.ticker}"
            x = float(pct_e.get(r.ticker, 0.5))
            graus = float(np.degrees(a)) % 360
            esq = 90 < graus < 270
            proprio = comp_fat["pct_especifico"].get(r.ticker) if comp_fat is not None else None
            nos.append({"id": i, "name": r.ticker, "value": _polar(R_ABERTO if abre else R_EMP, a), "category": 2,
                        "symbolSize": float((9 if abre else 5) + (8 if abre else 5) * x),
                        "itemStyle": {"color": cor_cen(r.ticker, _cor(x)), "borderColor": TXT,
                                      "borderWidth": 2 if r.ticker in carteira else 0, "opacity": opac(i, 1.0 if abre or aceso else 0.75) if not aberto or abre else 0.2},
                        "label": {"show": abre or r.ticker in carteira or (aceso is not None and i in aceso
                                                                            and (foco in SECTORS or esc is not None)),
                                  **({"formatter": f"{r.ticker} {fm.pct(cenario[r.ticker], 1, sinal=True)}"}
                                     if esc and r.ticker in cenario.index else {}),
                                  "color": TXT, "fontSize": 10, "position": "left" if esq else "right",
                                  "rotate": graus - 180 if esq else graus, "opacity": opac(i)},
                        "dica": f"<b>{r.ticker}</b> · {r.nome}<br/>" + (f"{fm.pct(r.peso)} do {s}" if r.peso else f"setor {s}")
                                + (f"<br/>ES 97,5% previsto {fm.pct(es_e[r.ticker])}" if r.ticker in es_e else "")
                                + (f"<br/>Risco próprio {fm.pct(proprio, 0)}" if proprio is not None and proprio == proprio else "")
                                + txt_cen(r.ticker)
                                + ("<br/>fora do mapa (acrescentada na Minha carteira)" if r.peso == 0 else "")
                                + "<br/><i>clica para abrir a ficha</i>"})
            arestas.append({"source": s, "target": i,
                            "lineStyle": {"color": COR_FIO, "width": 0.8 if abre else 0.5,
                                          "opacity": opac(i, 0.7 if abre else 0.35), "curveness": 0}})
    # setor ↔ setor (árvore das quedas conjuntas)
    if len(p_set):
        arv = cd.graph(p_set, by=por)
        for a, b, d in arv.edges(data=True):
            liga = aceso is None or {a, b} <= aceso
            dica = (f"{a} ↔ {b}: caem juntos em {fm.pct(d['p_conj'], 0)} dos {q_txt} piores dias" if por == "p_conj"
                    else f"{a} ↔ {b}: correlação {fm.num(d['corr'])}")
            arestas.append({"source": a, "target": b, "dica": dica,
                            "lineStyle": {"color": COR_CAUDA, "width": 1 + 9 * max(d[por] - (0.1 if por == "p_conj" else 0.5), 0),
                                          "opacity": 0.75 if liga else APAGADO, "curveness": 0.25}})
    # fator → setor
    bmax = lf["beta"].abs().max() if len(lf) else 1
    for r in lf.itertuples():
        ag = r.sinal == "+"
        liga = aceso is None or {r.fator, r.setor} <= aceso
        d_ = diretos.get((r.fator, r.setor))
        arestas.append({"source": r.fator, "target": r.setor,
                        **({"label": {"show": True, "formatter": fm.pct(d_, 1, sinal=True), "color": TXT, "fontSize": 10,
                                      "backgroundColor": "#0b1220", "padding": [1, 3], "borderRadius": 3}}
                           if d_ is not None and liga else {}),
                        "dica": f"{MACRO_FACTORS[r.fator]['curto']} a subir {'agrava' if ag else 'atenua'} o {r.setor}"
                                f"<br/>β padronizado {fm.num(r.beta, 2, sinal=True)} · t {fm.num(r.t, 1, sinal=True)}",
                        "lineStyle": {"color": RISCO["vermelho"] if ag else RISCO["verde"], "type": "solid" if ag else "dashed",
                                      "width": 1 + 4 * abs(r.beta) / bmax,
                                      # sem foco ficam discretas (o centro seria um novelo); acendem ao passar o rato
                                      "opacity": (0.75 if foco else 0.16) if liga else APAGADO,
                                      "curveness": 0.12}})
    # empresas do setor aberto: árvore das suas quedas conjuntas
    if aberto and pares is not None:
        pe = pares[pares["grupo"] == aberto]
        pe = pe[(pe["a"] != aberto) & (pe["b"] != aberto)]
        if len(pe):
            for a, b, d in cd.graph(pe, by=por).edges(data=True):
                arestas.append({"source": f"{aberto}/{a}", "target": f"{aberto}/{b}",
                                "dica": (f"{a} ↔ {b}: caem juntas em {fm.pct(d['p_conj'], 0)} dos piores dias" if por == "p_conj"
                                         else f"{a} ↔ {b}: correlação {fm.num(d['corr'])}"),
                                "lineStyle": {"color": COR_CAUDA, "width": 0.6 + 4 * max(d[por] - (0.1 if por == "p_conj" else 0.5), 0),
                                              "opacity": 0.6, "curveness": 0.4}})

    # --- o choque a correr pela cadeia (partículas por cima das ligações)
    pos = {n["id"]: n["value"] for n in nos}
    vagas: list[list[dict]] = [[], []]
    if foco in MACRO_FACTORS:
        for r in lf[lf["fator"] == foco].itertuples():
            cor = RISCO["vermelho"] if r.sinal == "+" else RISCO["verde"]
            vagas[0].append({"coords": [pos[foco], pos[r.setor]], "lineStyle": {"color": cor}})
            vagas[1] += [{"coords": [pos[r.setor], pos[i]], "lineStyle": {"color": cor}}
                         for i in pos if i.startswith(f"{r.setor}/")]
    elif foco in SECTORS:
        vagas[0] = [{"coords": [pos[f], pos[foco]], "lineStyle": {"color": COR_FATOR}}
                    for f in lf.loc[lf["setor"] == foco, "fator"]]
        vagas[1] = [{"coords": [pos[foco], pos[i]], "lineStyle": {"color": COR_CAUDA}}
                    for i in (aceso or set()) if i in pos and i not in MACRO_FACTORS and i != foco]
    efeitos = [{"type": "lines", "coordinateSystem": "cartesian2d", "zlevel": 2, "silent": True, "data": v,
                "lineStyle": {"width": 0, "curveness": 0.12},
                "effect": {"show": True, "period": 2.2, "delay": 900 * k, "trailLength": 0.35, "symbol": "circle",
                           "symbolSize": 5 if k == 0 else 3, "loop": True}}
               for k, v in enumerate(vagas) if v]

    eixo = {"show": False, "min": -LIM, "max": LIM, "type": "value"}
    return {
        "backgroundColor": "transparent",
        "animationDurationUpdate": 600,
        "grid": {"left": "center", "top": 36, "width": ALTURA - 46, "height": ALTURA - 46, "containLabel": False},
        "xAxis": eixo, "yAxis": eixo,
        "dataZoom": [{"type": "inside", "xAxisIndex": 0, "filterMode": "none"},
                     {"type": "inside", "yAxisIndex": 0, "filterMode": "none"}],
        "legend": {"data": ["Fatores macro", "Setores", "Empresas"], "top": 0, "textStyle": {"color": MUTED},
                   "itemGap": 18, "icon": "circle"},
        "tooltip": {"trigger": "item", "backgroundColor": "#131c2e", "borderColor": "#1e293b",
                    "textStyle": {"color": TXT, "fontSize": 12}, "formatter": JS_DICA},
        "series": [{"type": "graph", "layout": "none", "coordinateSystem": "cartesian2d", "data": nos, "links": arestas,
                    "categories": [{"name": "Fatores macro", "itemStyle": {"color": COR_FATOR}},
                                   {"name": "Setores", "itemStyle": {"color": RISCO["amarelo"]}},
                                   {"name": "Empresas", "itemStyle": {"color": MUTED}}],
                    "emphasis": {"focus": "adjacency", "lineStyle": {"opacity": 0.95}},
                    "blur": {"itemStyle": {"opacity": 0.12}, "lineStyle": {"opacity": 0.04}},
                    "lineStyle": {"opacity": 0.5}, "zlevel": 1}] + efeitos,
    }


def evento(valor) -> str | None:
    """Id do nó clicado no valor devolvido pelo componente ("id|instante"), ou None."""
    try:
        v = valor["chart_event"] if not isinstance(valor, str) else valor
    except (KeyError, TypeError):
        v = getattr(valor, "chart_event", None)
    return v if isinstance(v, str) and "|" in v else None


def desenhar(g: dict, aberto: str | None, q_txt: str, cenario: pd.Series | None = None,
             chocados: list[str] | None = None, por: str = "p_conj", carteira: set[str] | None = None,
             hold: pd.DataFrame | None = None, diretos: dict | None = None) -> str | None:
    """Desenha a teia e devolve o clique novo (id do nó), ou None. Com um cenário, a cadeia animada é a do 1.º fator
    macro chocado."""
    from streamlit_echarts import JsCode, st_echarts
    foco = next((f for f in (chocados or []) if f in MACRO_FACTORS), None) if cenario is not None else None
    op = opcoes(g["sc"], g["links"], g["cauda"], g["hold"] if hold is None else hold, g["comp_risk"], g["comp_fat"], aberto,
                foco or st.session_state.get("teia_foco"), q_txt, cenario, por, carteira, diretos)
    op["tooltip"]["formatter"] = JsCode(JS_DICA)
    res = st_echarts(op, events={"click": JS_CLIQUE}, height=f"{ALTURA}px", key="teia_chart", theme="dark")
    clique = evento(res)
    if clique and clique != st.session_state.get("teia_ult"):
        st.session_state["teia_ult"] = clique
        return clique.split("|")[0]
    return None
