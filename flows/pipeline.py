"""Flow Prefect do pipeline diário: uma task por bot.

Coletor → Inspetor → Modelador → Backtester → Carteira → Scorecard → Ligações → Repórter → Sentinela → Relatório → Snapshot
- Coletor: retries=COLLECT_RETRIES (rede); se os preços vierem atrasados, repete o download até
  COLLECT_STALE_TRIES vezes. Cache pela última sessão de NY fechada, e só com os dados já completos.
- As tasks são chamadas em sequência: se uma falhar (ex.: o contrato do Inspetor), a exceção pára o flow
  e nada a jusante corre.

Uso:  .venv/bin/python -m flows.pipeline                 (tudo, com download)
      .venv/bin/python -m flows.pipeline --offline       (usa data/raw existente)
      .venv/bin/python -m flows.pipeline --semanal       (força o relatório semanal)
      .venv/bin/python -m flows.pipeline --sem-notificacao
Agendamento: flows/com.fernando.sectorrisk.plist (launchd, dias úteis às 22:30), instalado com flows/install_launchd.sh.
"""
import os
import sys
import time
from datetime import date
from pathlib import Path

# Sem telemetria do Prefect: não envia dados para fora e evita o aviso "database is locked" no servidor temporário.
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

import pandas as pd  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import (COLLECT_RETRIES, COLLECT_RETRY_DELAY, COLLECT_STALE_TRIES,  # noqa: E402
                             COLLECT_STALE_WAIT, RAW)
from src.clean import validate  # noqa: E402
from src.collect import download_factors, download_prices  # noqa: E402
from src.collect.sessao import dados_completos, ultima_sessao, ultimo_fecho  # noqa: E402
from src.models import backtest, factor_links, portfolio, risk_metrics, scorecard  # noqa: E402
from src.report import alerts, build_map, snapshot, weekly  # noqa: E402


def _ultimo_fecho_raw() -> date:
    return ultimo_fecho(pd.read_parquet(RAW / "prices.parquet"))


def _cache_por_sessao(ctx, params) -> str | None:
    """Chave de cache = última sessão de NY já fechada (não a data do calendário: um run de madrugada em
    Lisboa ainda espera a sessão da véspera). Sem cache se os ficheiros brutos tiverem desaparecido ou se ainda
    não chegarem a essa sessão: assim um run nunca fica preso a uma coleta com dados velhos."""
    if not all((RAW / f).exists() for f in ("prices.parquet", "factors.parquet")):
        return None
    if not dados_completos(_ultimo_fecho_raw(), date.fromisoformat(params["sessao"])):
        return None
    return f"coletor-sessao-{params['sessao']}"


@task(name="Coletor", retries=COLLECT_RETRIES, retry_delay_seconds=COLLECT_RETRY_DELAY,
      cache_key_fn=_cache_por_sessao, persist_result=True)
def coletor(sessao: str) -> str:
    """Descarrega preços e fatores. Se os preços ainda não chegarem à sessão esperada, repete até
    COLLECT_STALE_TRIES vezes; depois segue (o Sentinela alerta). Erros de rede usam os retries do Prefect."""
    log = get_run_logger()
    alvo = date.fromisoformat(sessao)
    for tentativa in range(1, COLLECT_STALE_TRIES + 1):
        download_prices.main()
        download_factors.main()
        ultimo = _ultimo_fecho_raw()
        if dados_completos(ultimo, alvo):
            return sessao
        if tentativa < COLLECT_STALE_TRIES:
            log.warning(f"Preços só até {ultimo}, esperado {sessao}: nova tentativa {tentativa + 1}/"
                        f"{COLLECT_STALE_TRIES} daqui a {COLLECT_STALE_WAIT // 60} min")
            time.sleep(COLLECT_STALE_WAIT)
    log.warning(f"Preços continuam só até {ultimo} (esperado {sessao}); o Sentinela vai alertar")
    return sessao


@task(name="Inspetor")
def inspetor():
    validate.main()


@task(name="Modelador")
def modelador():
    risk_metrics.main()


@task(name="Backtester")
def backtester():
    backtest.main()


@task(name="Carteira")
def carteira():
    portfolio.main()


@task(name="Scorecard")
def score():
    scorecard.main()


@task(name="Ligações")
def ligacoes():
    factor_links.main()


@task(name="Repórter")
def reporter():
    build_map.main()


@task(name="Snapshot")
def snapshot_cockpit(offline: bool):
    """Atualiza app/cockpit/snapshot (o commit e o push ficam à escolha do utilizador)."""
    snapshot.main(offline=offline)


@task(name="Sentinela")
def sentinela(notificar: bool) -> list[dict]:
    return alerts.main(notify=notificar)


@task(name="Relatório semanal")
def relatorio(forcar: bool):
    out = weekly.main(force=forcar)
    return str(out) if out else None


@flow(name="risco-setorial", log_prints=True)
def pipeline(offline: bool = False, semanal: bool = False, notificar: bool = True):
    log = get_run_logger()
    if offline:
        log.info("Modo offline: sem coleta, usa data/raw existente")
    else:
        sessao = ultima_sessao().isoformat()
        log.info(f"Sessão de NY esperada: {sessao}")
        coletor(sessao)
    inspetor()
    modelador()
    backtester()
    carteira()
    score()
    ligacoes()
    reporter()
    al = sentinela(notificar)
    rel = relatorio(semanal)
    snapshot_cockpit(offline)              # por último: leva os alertas e o histórico do score de hoje
    log.info(f"Fim: {len(al)} alertas" + (f"; relatório semanal em {rel}" if rel else ""))
    return al


if __name__ == "__main__":
    a = sys.argv[1:]
    pipeline(offline="--offline" in a, semanal="--semanal" in a, notificar="--sem-notificacao" not in a)
