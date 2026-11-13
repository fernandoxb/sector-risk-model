"""Motor de cenários: um choque nos fatores macro propaga-se aos outros fatores, aos setores e às empresas (Fase 3).

Dados semanais (sexta a sexta), janela de CENARIO_ANOS anos. Fatores: os seis macro e o mercado (retorno log do SPY).
Três camadas, simuladas por Monte Carlo:
  1. fatores   os fatores não chocados seguem a distribuição normal condicional ao choque (média e covariância
               da janela): E[x_u | x_s] = h·μ_u + Σ_us Σ_ss⁻¹ (x_s − h·μ_s), Cov = h·(Σ_uu − Σ_us Σ_ss⁻¹ Σ_su)
  2. setores   r_s = h·a_s + B_s·x + ε_s, com B da regressão de cada setor nos seis fatores ao mesmo tempo (OLS) e
               ε a soma de h semanas reamostradas dos resíduos (bootstrap: guarda as caudas e a dependência entre
               setores, porque se sorteia a semana inteira)
  3. empresas  r_i = h·α_i + β_i·r_s + e_i, com β_i da regressão semanal na do seu ETF e e_i reamostrado nas mesmas
               semanas que os resíduos dos setores
Retornos log somados no horizonte de h semanas e devolvidos como retornos simples (e^r − 1).

Funções puras: semanal, ajustar, simular, resumo, episodio. I/O em main().
Saídas: data/features/cenario_semanal.parquet (os dados do cockpit) e cenario_validacao.csv (episódios históricos).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (CENARIO_ANOS, CENARIO_EPISODIOS, CENARIO_NSIM, BT_SEED, CLEAN, FEATURES,  # noqa: E402
                             MACRO_FACTORS, MARKET, SECTORS)
from src.collect.empresas import load_holdings  # noqa: E402
from src.models.factor_links import factor_changes, sector_returns, weekly_cut, weekly_levels  # noqa: E402

MERCADO = "mercado"
FAT = list(MACRO_FACTORS) + [MERCADO]   # o retorno log do SPY é o 7.º fator: é por ele que as crises chegam aos setores


def semanal(etfs: pd.DataFrame, emp: pd.DataFrame | None, fatores: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por semana: variação de cada fator e retorno log de cada setor e empresa (colunas pelo ticker)."""
    fim = weekly_cut(etfs.index.max())
    precos = etfs if emp is None else etfs.join(emp[[c for c in emp.columns if c not in etfs.columns]], how="left")
    r = sector_returns(weekly_levels(precos.loc[:fim]))
    x = factor_changes(weekly_levels(fatores.loc[:fim])).assign(**{MERCADO: r[MARKET]})
    return x.join(r.drop(columns=MARKET), how="inner").iloc[1:]


def _preenche(res: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Semanas sem dados (a série ainda não existia) recebem resíduos sorteados da própria série."""
    res = res.copy()
    for j in range(res.shape[1]):
        falta = np.isnan(res[:, j])
        if falta.any() and (~falta).any():
            res[falta, j] = rng.choice(res[~falta, j], size=int(falta.sum()))
    return np.nan_to_num(res)


def ajustar(dados: pd.DataFrame, hold: pd.DataFrame, setores: list[str] = list(SECTORS),
            seed: int = BT_SEED) -> dict:
    """Parâmetros das três camadas, estimados em `dados` (saída de `semanal`)."""
    x = dados[FAT]
    ok = x.notna().all(axis=1)
    x = x[ok]
    d = dados[ok]
    setores = [s for s in setores if s in d and d[s].notna().sum() > 2 * len(FAT)]
    X1 = np.column_stack([np.ones(len(x)), x.to_numpy()])
    coef, res_s = {}, pd.DataFrame(np.nan, index=d.index, columns=setores)
    for s in setores:
        m = d[s].notna().to_numpy()
        b = np.linalg.lstsq(X1[m], d[s].to_numpy()[m], rcond=None)[0]
        coef[s] = b
        res_s.loc[m, s] = d[s].to_numpy()[m] - X1[m] @ b
    emp = [(h.ticker, h.setor) for h in hold.drop_duplicates("ticker").itertuples()
           if h.ticker in d and h.setor in setores and d[h.ticker].notna().sum() > 52]
    alfa, beta, res_e = {}, {}, pd.DataFrame(np.nan, index=d.index, columns=[t for t, _ in emp])
    for t, s in emp:
        m = (d[t].notna() & d[s].notna()).to_numpy()
        A = np.column_stack([np.ones(m.sum()), d[s].to_numpy()[m]])
        a_, b_ = np.linalg.lstsq(A, d[t].to_numpy()[m], rcond=None)[0]
        alfa[t], beta[t] = a_, b_
        res_e.loc[m, t] = d[t].to_numpy()[m] - A @ [a_, b_]
    rng = np.random.default_rng(seed)
    return {"mu": x.mean().to_numpy(), "sigma": x.cov().to_numpy(), "setores": setores,
            "a": np.array([coef[s][0] for s in setores]), "B": np.array([coef[s][1:] for s in setores]),
            "res_s": _preenche(res_s.to_numpy(), rng), "empresas": [t for t, _ in emp],
            "setor_de": dict(emp), "alfa": np.array([alfa[t] for t, _ in emp]), "beta": np.array([beta[t] for t, _ in emp]),
            "res_e": _preenche(res_e.to_numpy(), rng), "semanas": len(d), "inicio": d.index[0], "fim": d.index[-1]}


def simular(mod: dict, choque: dict[str, float], h: int = 1, n: int = CENARIO_NSIM, seed: int = BT_SEED) -> pd.DataFrame:
    """`n` cenários de h semanas. `choque`: fator → variação total no horizonte, na unidade da regressão (p.p. ou
    Δ log). Devolve retornos simples, uma coluna por fator (as variações), setor e empresa."""
    rng = np.random.default_rng(seed)
    s_idx = [FAT.index(f) for f in choque]
    u_idx = [i for i in range(len(FAT)) if i not in s_idx]
    mu, S = mod["mu"] * h, mod["sigma"] * h
    x = np.empty((n, len(FAT)))
    xs = np.array([choque[f] for f in choque], dtype=float)
    x[:, s_idx] = xs
    if u_idx:
        if s_idx:
            K = S[np.ix_(u_idx, s_idx)] @ np.linalg.inv(S[np.ix_(s_idx, s_idx)])
            m_u = mu[u_idx] + K @ (xs - mu[s_idx])
            C_u = S[np.ix_(u_idx, u_idx)] - K @ S[np.ix_(s_idx, u_idx)]
        else:
            m_u, C_u = mu[u_idx], S[np.ix_(u_idx, u_idx)]
        x[:, u_idx] = rng.multivariate_normal(m_u, (C_u + C_u.T) / 2, size=n, method="eigh")
    semanas = rng.integers(0, len(mod["res_s"]), size=(n, h))           # as mesmas semanas para setores e empresas
    r_s = h * mod["a"] + x @ mod["B"].T + mod["res_s"][semanas].sum(axis=1)
    pos = {s: k for k, s in enumerate(mod["setores"])}
    col_s = [pos[mod["setor_de"][t]] for t in mod["empresas"]]
    r_e = h * mod["alfa"] + mod["beta"] * r_s[:, col_s] + mod["res_e"][semanas].sum(axis=1)
    return pd.concat([pd.DataFrame(x, columns=FAT), pd.DataFrame(np.expm1(r_s), columns=mod["setores"]),
                      pd.DataFrame(np.expm1(r_e), columns=mod["empresas"])], axis=1)


def resumo(sims: pd.DataFrame, colunas: list[str] | None = None) -> pd.DataFrame:
    """Por série: mediana, os 5% piores (quantil 5%), o quantil 95% e a probabilidade de perder."""
    s = sims[colunas] if colunas else sims
    return pd.DataFrame({"mediana": s.median(), "q05": s.quantile(0.05), "q95": s.quantile(0.95), "p_perda": (s < 0).mean()})


def episodio(dados: pd.DataFrame, hold: pd.DataFrame, inicio: str, fim: str, fatores: list[str] = FAT,
             anos: int = CENARIO_ANOS, n: int = CENARIO_NSIM, seed: int = BT_SEED) -> pd.DataFrame:
    """Validação fora da amostra: estima o modelo só com as semanas antes de `inicio` (até `anos` anos), impõe as
    variações reais de `fatores` entre `inicio` e `fim` (os outros ficam condicionais) e compara a previsão de cada
    setor com o que aconteceu."""
    ini, fim_ = pd.Timestamp(inicio), pd.Timestamp(fim)
    antes = dados[(dados.index <= ini) & (dados.index > ini - pd.DateOffset(years=anos))]
    dentro = dados[(dados.index > ini) & (dados.index <= fim_)]
    mod = ajustar(antes, hold.iloc[:0], seed=seed)
    choque = dentro[fatores].sum().to_dict()
    res = resumo(simular(mod, choque, h=len(dentro), n=n, seed=seed), mod["setores"])
    real = np.expm1(dentro[mod["setores"]].sum())
    return res.assign(real=real, dentro_90=(real >= res["q05"]) & (real <= res["q95"]),
                      semanas=len(dentro), semanas_estimacao=len(antes)).rename_axis("setor").reset_index()


def main() -> Path:
    etfs = pd.read_parquet(CLEAN / "prices.parquet")
    emp_f = CLEAN / "prices_empresas.parquet"
    emp = pd.read_parquet(emp_f) if emp_f.exists() else None
    dados = semanal(etfs, emp, pd.read_parquet(CLEAN / "factors.parquet"))
    out = FEATURES / "cenario_semanal.parquet"
    janela = dados[dados.index > dados.index.max() - pd.DateOffset(years=CENARIO_ANOS)]
    janela.astype("float32").copy().rename_axis("data").reset_index().to_parquet(out, compression="zstd")   # data em coluna: o snapshot não guarda índices
    hold = load_holdings()
    condicoes = {"macro": list(MACRO_FACTORS), "macro_mercado": FAT}       # o mercado previsto ou o mercado real
    val = pd.concat([episodio(dados, hold, a, b, fat).assign(episodio=nome, condicao=c)
                     for nome, (a, b) in CENARIO_EPISODIOS.items() for c, fat in condicoes.items()], ignore_index=True)
    val.to_csv(FEATURES / "cenario_validacao.csv", index=False, float_format="%.6f")
    for (nome, c), v in val.groupby(["episodio", "condicao"], sort=False):
        print(f"[cenários] {nome} ({c}): {int(v['dentro_90'].sum())}/{len(v)} setores dentro do intervalo de 90%; "
              f"correlação previsto × real {v['mediana'].corr(v['real']):.2f}; "
              f"erro médio {(v['mediana'] - v['real']).abs().mean():.1%}, viés {(v['mediana'] - v['real']).mean():+.1%} "
              f"({int(v['semanas'].iloc[0])} semanas)")
    return out


if __name__ == "__main__":
    main()
