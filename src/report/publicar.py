"""Bot Publicador: commit e push semanal (última sessão da semana) só do snapshot do cockpit (app/cockpit/snapshot/).

Regras de segurança:
  - só faz commit dos ficheiros do snapshot (`git commit --only -- <pasta>`), mesmo que haja outras alterações;
  - só no ramo SNAPSHOT_PUBLISH_BRANCH e só se o snapshot mudou;
  - se o push falhar (ex.: o remoto avançou), o commit fica local e o erro volta para o flow, que alerta.
A decisão "é dia de publicar?" é pura (deve_publicar); o resto chama o git.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (ROOT, SNAPSHOT, SNAPSHOT_PUBLISH, SNAPSHOT_PUBLISH_BRANCH,  # noqa: E402
                             SNAPSHOT_PUBLISH_REMOTE)
from src.report.weekly import is_report_day  # noqa: E402


def deve_publicar(asof: str, ativo: bool = SNAPSHOT_PUBLISH, forcar: bool = False) -> bool:
    """Publica na última sessão da semana (normalmente sexta), ou sempre que for forçado."""
    return ativo and (forcar or is_report_day(asof))


def mensagem(asof: str) -> str:
    return f"Snapshot do cockpit: dados até {asof}\n\nPublicado automaticamente pelo bot Publicador (flows/pipeline.py)."


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} falhou: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


def publicar(repo: Path, pasta: Path, asof: str, ramo: str = SNAPSHOT_PUBLISH_BRANCH,
             remoto: str = SNAPSHOT_PUBLISH_REMOTE, push: bool = True) -> str | None:
    """Commit (e push) só de `pasta`. Devolve o hash do commit, ou None se o snapshot não mudou."""
    rel = str(pasta.resolve().relative_to(repo.resolve()))
    atual = _git(repo, "branch", "--show-current")
    if atual != ramo:
        raise RuntimeError(f"no ramo '{atual}', não em '{ramo}': snapshot não publicado")
    if not _git(repo, "status", "--porcelain", "--", rel):
        return None
    _git(repo, "add", "--", rel)
    _git(repo, "commit", "--only", "-m", mensagem(asof), "--", rel)
    sha = _git(repo, "rev-parse", "--short", "HEAD")
    if push:
        _git(repo, "push", remoto, ramo)
    return sha


def main(forcar: bool = False) -> str | None:
    asof = json.loads((SNAPSHOT / "meta.json").read_text())["dados_ate"]
    if not deve_publicar(asof, forcar=forcar):
        print(f"[publicador] {asof} não é sexta: snapshot não publicado (usar --publicar para forçar)")
        return None
    sha = publicar(ROOT, SNAPSHOT, asof)
    print(f"[publicador] {'commit ' + sha + ' enviado para ' + SNAPSHOT_PUBLISH_REMOTE if sha else 'sem alterações'}")
    return sha


if __name__ == "__main__":
    main(forcar="--publicar" in sys.argv)
