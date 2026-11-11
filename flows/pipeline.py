"""Flow Prefect do pipeline diário: uma task por bot.

Coletor → Inspetor → Modelador → Backtester → Carteira → Scorecard → Ligações → Repórter → Snapshot → Sentinela → Relatório
- Coletor: retries=COLLECT_RETRIES (rede) e cache por data: dois runs no mesmo dia descarregam uma só vez.
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
from datetime import date
from pathlib import Path

# Sem telemetria do Prefect: não envia dados para fora e evita o aviso "database is locked" no servidor temporário.
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

from prefect import flow, get_run_logger, task  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import COLLECT_RETRIES, COLLECT_RETRY_DELAY, RAW  # noqa: E402
from src.clean import validate  # noqa: E402
from src.collect import download_factors, download_prices  # noqa: E402
from src.models import backtest, factor_links, portfolio, risk_metrics, scorecard  # noqa: E402
from src.report import alerts, build_map, snapshot, weekly  # noqa: E402


def _cache_por_dia(ctx, params) -> str | None:
    """Chave de cache = data do run; sem cache se os ficheiros brutos tiverem desaparecido."""
    if not all((RAW / f).exists() for f in ("prices.parquet", "factors.parquet")):
        return None
    return f"coletor-{params['dia']}"


@task(name="Coletor", retries=COLLECT_RETRIES, retry_delay_seconds=COLLECT_RETRY_DELAY,
      cache_key_fn=_cache_por_dia, persist_result=True)
def coletor(dia: str) -> str:
    download_prices.main()
    download_factors.main()
    return dia


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
        coletor(date.today().isoformat())
    inspetor()
    modelador()
    backtester()
    carteira()
    score()
    ligacoes()
    reporter()
    snapshot_cockpit(offline)
    al = sentinela(notificar)
    rel = relatorio(semanal)
    log.info(f"Fim: {len(al)} alertas" + (f"; relatório semanal em {rel}" if rel else ""))
    return al


if __name__ == "__main__":
    a = sys.argv[1:]
    pipeline(offline="--offline" in a, semanal="--semanal" in a, notificar="--sem-notificacao" not in a)
