"""Flow Prefect do pipeline diário: uma task por bot.

Coletor → Inspetor → Modelador → Backtester → Backtester de empresas → Carteira → Scorecard → Ligações → Cauda → Sentinela → Relatório →
Snapshot → Publicador (às sextas: envia data/snapshot/ para o ramo snapshot, num único commit)
- Coletor: retries=COLLECT_RETRIES (rede); se os preços vierem atrasados, repete o download até
  COLLECT_STALE_TRIES vezes. Cache pela última sessão de NY fechada, e só com os dados já completos.
- As tasks são chamadas em sequência: se uma falhar (ex.: o contrato do Inspetor), a exceção pára o flow
  e nada a jusante corre.

Uso:  .venv/bin/python -m flows.pipeline                 (tudo, com download)
      .venv/bin/python -m flows.pipeline --offline       (usa data/raw existente)
      .venv/bin/python -m flows.pipeline --semanal       (força o relatório semanal)
      .venv/bin/python -m flows.pipeline --sem-notificacao
      .venv/bin/python -m flows.pipeline --publicar        (publica o snapshot mesmo que não seja sexta)
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
from src.collect import download_factors, download_prices, empresas  # noqa: E402
from src.collect.sessao import dados_completos, ultima_sessao, ultimo_fecho  # noqa: E402
from src.models import (backtest, backtest_empresas, cauda, cenarios, decomposicao, factor_links, horizonte, portfolio,  # noqa: E402
                        risk_metrics, scorecard)
from src.report import alerts, publicar, snapshot, weekly  # noqa: E402


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
        empresas.main()
        empresas.main_prices()
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
    horizonte.main()


@task(name="Backtester de empresas")
def backtester_empresas():
    backtest_empresas.main()


@task(name="Carteira")
def carteira():
    portfolio.main()


@task(name="Scorecard")
def score():
    scorecard.main()


@task(name="Ligações")
def ligacoes():
    factor_links.main()
    decomposicao.main()
    cenarios.main()


@task(name="Cauda")
def dependencia_cauda():
    cauda.main()


@task(name="Snapshot")
def snapshot_cockpit(offline: bool, fichas: bool = False):
    """Grava data/snapshot (fora do Git); o Publicador envia-o para o ramo snapshot."""
    snapshot.main(offline=offline, fichas=fichas)


@task(name="Publicador")
def publicador(forcar: bool, notificar: bool):
    """Envio semanal do snapshot para o ramo snapshot. Uma falha (ex.: sem rede) não estraga o run: fica no log e notifica."""
    log = get_run_logger()
    try:
        publicar.main(forcar=forcar)
    except Exception as e:  # noqa: BLE001
        log.error(f"Snapshot não publicado: {e}")
        if notificar:
            alerts.send_notification(f"Snapshot do cockpit não publicado: {e}"[:200])


@task(name="Sentinela")
def sentinela(notificar: bool) -> list[dict]:
    return alerts.main(notify=notificar)


@task(name="Relatório semanal")
def relatorio(forcar: bool):
    out = weekly.main(force=forcar)
    return str(out) if out else None


@flow(name="risco-setorial", log_prints=True)
def pipeline(offline: bool = False, semanal: bool = False, notificar: bool = True, publicar_snapshot: bool = False):
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
    backtester_empresas()
    carteira()
    score()
    ligacoes()
    dependencia_cauda()
    al = sentinela(notificar)
    rel = relatorio(semanal)
    snapshot_cockpit(offline, publicar_snapshot)              # depois do Sentinela: leva os alertas e o histórico do score de hoje
    if not offline:
        publicador(publicar_snapshot, notificar)
    log.info(f"Fim: {len(al)} alertas" + (f"; relatório semanal em {rel}" if rel else ""))
    return al


if __name__ == "__main__":
    a = sys.argv[1:]
    pipeline(offline="--offline" in a, semanal="--semanal" in a, notificar="--sem-notificacao" not in a,
             publicar_snapshot="--publicar" in a)
