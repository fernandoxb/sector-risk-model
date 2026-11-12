"""Glossário dos termos do cockpit.

Gerado a partir do antigo app.py: o código do separador é o mesmo; os dados do run chegam em `g`.
"""
import streamlit as st
from config.settings import BT_TL_DAYS, CVAR_YEARS, PANIC_Q, PI_LEVELS, RATE_LOOKBACK, RATE_THRESHOLD_PP
from src.report import formato as fm
from paineis.comum import N_CVAR, OFICIAL


def render(g: dict) -> None:
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
