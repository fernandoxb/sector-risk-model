"""Dependência de cauda entre setores e entre empresas: quem cai com quem nos piores dias.

A correlação linear mede o andamento conjunto em dias normais e subestima as quedas conjuntas: na cópula de
Gauss a dependência de cauda é zero para qualquer correlação < 1 (McNeil, Frey e Embrechts 2005, ex. 5.32).
Aqui medem-se duas coisas:

  joint_crash    frequência observada de quedas conjuntas: P(B na cauda | A na cauda), com a cauda = q piores dias
                 de cada série (simétrica, porque as duas caudas têm a mesma probabilidade q; sob independência = q)
  cópula t       ρ_ij = sin(π τ_ij / 2) pelo tau de Kendall (secção 5.5.1); ν comum por máxima verosimilhança
                 nas pseudo-observações (5.5.2–5.5.3); λ_ij = 2 t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ)))   (eq. 5.31)

A cópula é ajustada aos resíduos padronizados de um GJR-GARCH(1,1)-t por série, para tirar os clusters de
volatilidade (os resíduos são aproximadamente iid; os retornos não). A frequência observada usa os retornos
simples, que são o que o investidor sente.

Funções puras: pseudo_obs, tau_to_rho, nearest_corr, lambda_t, t_copula_loglik, fit_nu, joint_crash, fit_group,
pairs_table, graph. Com GARCH: garch_residuals. I/O em main().
Cenários conjuntos (Fase D): simulate_t_copula, scenarios.
Janelas (CAUDA_JANELAS, 1/3/5/10 anos): fit_windows. A de CVAR_YEARS vai para tail_dependence.csv; as outras para
tail_dependence_janelas.csv (coluna `anos`).
Saídas: data/features/tail_dependence.csv (pares), tail_meta.json (ν e nº de dias por grupo e por janela) e
tail_scenarios.csv (setores: o que acontece aos outros quando um tem um dia mau).
"""
import json
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (BT_SEED, CAUDA_JANELAS, CAUDA_MIN_COBERTURA, CAUDA_NSIM, CAUDA_NU_BOUNDS, CAUDA_Q, CLEAN, CVAR_YEARS, FEATURES,  # noqa: E402
                             SECTORS)
from src.collect.empresas import load_holdings  # noqa: E402

GRUPO_SETORES = "setores"


# ----------------------------------------------------------------------------- cópula t
def pseudo_obs(x: pd.DataFrame) -> pd.DataFrame:
    """Pseudo-observações da cópula: posto de cada valor na sua coluna a dividir por n + 1 (em (0, 1))."""
    return x.rank(axis=0) / (len(x) + 1)


def tau_to_rho(tau):
    """Correlação da cópula elíptica a partir do tau de Kendall: ρ = sin(π τ / 2)."""
    return np.sin(np.pi * np.asarray(tau, dtype=float) / 2)


def nearest_corr(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Matriz de correlação mais próxima no sentido simples: corta os valores próprios abaixo de `eps` e repõe a
    diagonal a 1. Necessário porque a matriz dos sin(π τ / 2) par a par pode não ser definida positiva."""
    p = (p + p.T) / 2
    vals, vecs = np.linalg.eigh(p)
    if vals.min() >= eps:
        return p
    q = vecs @ np.diag(np.clip(vals, eps, None)) @ vecs.T
    d = np.sqrt(np.diag(q))
    return q / np.outer(d, d)


def lambda_t(rho, nu: float):
    """Coeficiente de dependência de cauda da cópula t (igual nas duas caudas): 2 t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ)))."""
    rho = np.clip(np.asarray(rho, dtype=float), -0.999999, 1.0)
    return 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)


def t_copula_loglik(u: np.ndarray, p: np.ndarray, nu: float) -> float:
    """Log-verosimilhança da cópula t (ex. 5.59): Σ_t [ln g_{ν,P}(z_t) − Σ_j ln g_ν(z_tj)], com z = t_ν⁻¹(u)."""
    z = stats.t.ppf(u, nu)
    d = p.shape[0]
    sign, logdet = np.linalg.slogdet(p)
    quad = np.einsum("ti,ij,tj->t", z, np.linalg.inv(p), z)
    from scipy.special import gammaln
    joint = (gammaln((nu + d) / 2) - gammaln(nu / 2) - d / 2 * np.log(nu * np.pi) - logdet / 2
             - (nu + d) / 2 * np.log1p(quad / nu))
    return float(joint.sum() - stats.t.logpdf(z, nu).sum())


def fit_nu(u: np.ndarray, p: np.ndarray, bounds: tuple[float, float] = CAUDA_NU_BOUNDS) -> float:
    """ν que maximiza a verosimilhança da cópula t, com a matriz P fixa (estimada pelo tau de Kendall)."""
    res = optimize.minimize_scalar(lambda v: -t_copula_loglik(u, p, v), bounds=bounds, method="bounded")
    return float(res.x)


# ----------------------------------------------------------------------------- quedas conjuntas observadas
def joint_crash(x: pd.DataFrame, q: float = CAUDA_Q) -> pd.DataFrame:
    """P(B nos q piores dias | A nos q piores dias), para todos os pares. Diagonal = 1; independência ≈ q."""
    tail = (x.rank(axis=0, method="first") <= np.floor(q * len(x))).to_numpy(dtype=float)
    k = tail.sum(axis=0)[0]
    return pd.DataFrame(tail.T @ tail / k, index=x.columns, columns=x.columns)


# ----------------------------------------------------------------------------- um grupo de séries
def fit_group(rets: pd.DataFrame, resid: pd.DataFrame, q: float = CAUDA_Q) -> dict:
    """Tudo para um grupo (ex.: os 11 setores): correlação, tau, ρ da cópula, ν, λ e quedas conjuntas.

    rets:  retornos simples (dias comuns)      resid: resíduos padronizados do GARCH, mesmas colunas
    """
    resid = resid.dropna()
    tau = resid.corr(method="kendall")
    p = nearest_corr(tau_to_rho(tau.to_numpy()))
    nu = fit_nu(pseudo_obs(resid).to_numpy(), p)
    cols = list(resid.columns)
    return {"corr": rets.corr(), "tau": tau, "rho_t": pd.DataFrame(p, index=cols, columns=cols), "nu": nu,
            "lam": pd.DataFrame(lambda_t(p, nu), index=cols, columns=cols), "p_conj": joint_crash(rets.dropna(), q),
            "n": len(resid), "rets": rets.dropna()}


# ----------------------------------------------------------------------------- cenários conjuntos (Fase D)
def simulate_t_copula(p: np.ndarray, nu: float, n: int = CAUDA_NSIM, seed: int = BT_SEED) -> np.ndarray:
    """`n` dias simulados da cópula t com matriz ρ e ν graus de liberdade, em uniformes (QRM, algoritmo 5.10):
    X = Z / √(W/ν), com Z ~ N(0, ρ) e W ~ χ²_ν; U = t_ν(X)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, len(p))) @ np.linalg.cholesky(p).T
    return stats.t.cdf(z / np.sqrt(rng.chisquare(nu, (n, 1)) / nu), nu)


def scenarios(rets: pd.DataFrame, p: np.ndarray, nu: float, q: float = CAUDA_Q, n: int = CAUDA_NSIM,
              seed: int = BT_SEED) -> pd.DataFrame:
    """"Se A tiver um dia mau (nos seus q piores), o que acontece a B?" Uma linha por par ordenado (cenario, serie).

    Os uniformes da cópula passam a retornos pelos quantis observados de cada série (margens empíricas).
      p_cauda      P(B também nos seus q piores dias | A nos seus q piores)
      perda_media  perda média de B nesses dias (positiva = perda); na linha cenario == serie, a do próprio A
      perda_normal perda média de B em todos os dias simulados, para comparar
      p_obs, perda_obs  as mesmas duas medidas contadas nos dias reais (poucos: q × nº de dias), para confrontar
    """
    cols = list(rets.columns)
    u = simulate_t_copula(np.asarray(p, dtype=float), nu, n, seed)
    x = np.column_stack([np.quantile(rets[c].to_numpy(), u[:, j]) for j, c in enumerate(cols)])
    jc = joint_crash(rets, q)
    real = (rets.rank(axis=0, method="first") <= np.floor(q * len(rets))).to_numpy()
    rows = []
    for i, a in enumerate(cols):
        mau = u[:, i] <= q
        for j, b in enumerate(cols):
            rows.append({"cenario": a, "serie": b, "p_cauda": float((u[mau, j] <= q).mean()),
                         "perda_media": float(-x[mau, j].mean()), "perda_normal": float(-x[:, j].mean()),
                         "p_obs": float(jc.iloc[i, j]), "perda_obs": float(-rets.iloc[real[:, i], j].mean())})
    return pd.DataFrame(rows)


def pairs_table(fit: dict, grupo: str) -> pd.DataFrame:
    """Formato longo, um par por linha (a < b na ordem das colunas)."""
    cols = list(fit["corr"].columns)
    rows = [{"grupo": grupo, "a": a, "b": b, **{k: float(fit[k].loc[a, b]) for k in ("corr", "tau", "rho_t", "lam", "p_conj")}}
            for i, a in enumerate(cols) for b in cols[i + 1:]]
    return pd.DataFrame(rows)


def graph(pairs: pd.DataFrame, by: str = "p_conj") -> nx.Graph:
    """Árvore geradora mínima das ligações mais fortes segundo `by` (distância = 1 − semelhança).
    Cada aresta leva corr, lam e p_conj, para o mapa mostrar as três."""
    g = nx.Graph()
    for r in pairs.itertuples():
        g.add_edge(r.a, r.b, weight=1 - getattr(r, by), corr=r.corr, lam=r.lam, p_conj=r.p_conj)
    return nx.minimum_spanning_tree(g, weight="weight")


# ----------------------------------------------------------------------------- GARCH
def garch_residuals(r: pd.Series) -> pd.Series:
    """Resíduos padronizados de um GJR-GARCH(1,1)-t ajustado aos retornos log (em %) da série."""
    x = (np.log1p(r.dropna()) * 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(x, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
    return (res.resid / res.conditional_volatility).rename(r.name)


def _window(prices: pd.DataFrame, years: int = CVAR_YEARS) -> pd.DataFrame:
    return prices[prices.index > prices.index.max() - pd.DateOffset(years=years)]


def _fit(prices: pd.DataFrame, cols: list[str], anos: int = CVAR_YEARS, q: float = CAUDA_Q,
         cobertura: float = CAUDA_MIN_COBERTURA) -> dict | None:
    """Um grupo numa janela de `anos`: entram as séries com dados em pelo menos `cobertura` dos dias da janela, e as
    contas usam os dias em que todas têm preço (na janela de 10 anos, os setores começam com o XLC, em 2018)."""
    rets = _window(prices[cols], anos).pct_change().iloc[1:]
    cols = [c for c in cols if rets[c].notna().mean() >= cobertura]
    rets = rets[cols].dropna()
    if len(cols) < 2:
        return None
    resid = pd.concat([garch_residuals(rets[c]) for c in cols], axis=1)
    return fit_group(rets, resid, q)


def fit_windows(etfs: pd.DataFrame, emp: pd.DataFrame | None, hold: pd.DataFrame | None, anos: int,
                q: float) -> dict[str, dict]:
    """Os setores e, se houver preços das empresas, um grupo por setor (as suas empresas + o próprio ETF)."""
    fits = {GRUPO_SETORES: _fit(etfs, list(SECTORS), anos, q)}
    if emp is not None:
        for s in SECTORS:
            tk = [t for t in hold.loc[hold["setor"] == s, "ticker"] if t in emp.columns]
            fits[s] = _fit(emp[tk].join(etfs[s], how="inner"), tk + [s], anos, q) if tk else None
    return {g: f for g, f in fits.items() if f}


def main() -> Path:
    etfs = pd.read_parquet(CLEAN / "prices.parquet")[list(SECTORS)]
    emp_f = CLEAN / "prices_empresas.parquet"
    emp, hold = (pd.read_parquet(emp_f), load_holdings()) if emp_f.exists() else (None, None)
    por_janela = {anos: fit_windows(etfs, emp, hold, anos, q) for anos, q in CAUDA_JANELAS.items()}
    fits = por_janela[CVAR_YEARS]

    def pares(fs: dict) -> pd.DataFrame:
        return pd.concat([pairs_table(f, g) for g, f in fs.items()], ignore_index=True)

    def resumo(fs: dict) -> dict:
        return {g: {"nu": round(f["nu"], 2), "dias": f["n"], "series": len(f["corr"])} for g, f in fs.items()}

    FEATURES.mkdir(parents=True, exist_ok=True)
    out = FEATURES / "tail_dependence.csv"                    # a janela padrão, no formato de sempre
    pares(fits).to_csv(out, index=False, float_format="%.6f")
    pd.concat([pares(fs).assign(anos=a) for a, fs in por_janela.items() if a != CVAR_YEARS], ignore_index=True).to_csv(
        FEATURES / "tail_dependence_janelas.csv", index=False, float_format="%.6f")
    meta = {"q": CAUDA_Q, "anos": CVAR_YEARS, "grupos": resumo(fits),
            "janelas": {str(a): {"q": CAUDA_JANELAS[a], "grupos": resumo(fs)} for a, fs in por_janela.items()}}
    (FEATURES / "tail_meta.json").write_text(json.dumps(meta, indent=1))
    s = fits[GRUPO_SETORES]
    scenarios(s["rets"], s["rho_t"].to_numpy(), s["nu"]).to_csv(FEATURES / "tail_scenarios.csv", index=False,
                                                                  float_format="%.6f")
    top6 = pairs_table(s, GRUPO_SETORES).sort_values("p_conj", ascending=False)
    print(f"[cauda] {len(fits)} grupos; setores: ν = {s['nu']:.1f}, {s['n']} dias; quedas conjuntas (q = {CAUDA_Q:.0%}):")
    print(top6.head(6)[["a", "b", "corr", "lam", "p_conj"]].round(3).to_string(index=False))
    for a, fs in por_janela.items():
        st_ = fs[GRUPO_SETORES]
        top = pairs_table(st_, GRUPO_SETORES).nlargest(1, "p_conj").iloc[0]
        print(f"[cauda] {a} anos (q = {CAUDA_JANELAS[a]:.0%}): {len(fs)} grupos; setores {st_['n']} dias, ν = {st_['nu']:.1f}; "
              f"par mais ligado {top.a}–{top.b} {top.p_conj:.0%}")
    return out


if __name__ == "__main__":
    main()
