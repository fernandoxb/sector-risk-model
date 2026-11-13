"""Bot Publicador: na última sessão da semana, envia o snapshot do cockpit para o ramo SNAPSHOT_PUBLISH_BRANCH.

Regras de segurança:
  - o ramo leva um único commit sem pai, só com os ficheiros do snapshot, que substitui o anterior (push forçado
    só desse ramo); o ramo de trabalho, o índice e os ficheiros locais não são tocados;
  - recusa publicar no ramo em que se está a trabalhar (nunca reescreve o `main`);
  - sem alterações no snapshot não há commit; se o push falhar, o erro volta para o flow, que alerta.
A decisão "é dia de publicar?" é pura (deve_publicar); o resto chama o git.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (ROOT, SNAPSHOT_LIVE, SNAPSHOT_PUBLISH, SNAPSHOT_PUBLISH_BRANCH,  # noqa: E402
                             SNAPSHOT_PUBLISH_REMOTE)
from src.report.weekly import is_report_day  # noqa: E402


def deve_publicar(asof: str, ativo: bool = SNAPSHOT_PUBLISH, forcar: bool = False) -> bool:
    """Publica na última sessão da semana (normalmente sexta), ou sempre que for forçado."""
    return ativo and (forcar or is_report_day(asof))


def mensagem(asof: str) -> str:
    return f"Snapshot do cockpit: dados até {asof}\n\nPublicado automaticamente pelo bot Publicador (flows/pipeline.py)."


def _git(repo: Path, *args: str, env: dict | None = None) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                       env=None if env is None else os.environ | env)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} falhou: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


def publicar(repo: Path, pasta: Path, asof: str, ramo: str = SNAPSHOT_PUBLISH_BRANCH,
             remoto: str = SNAPSHOT_PUBLISH_REMOTE, push: bool = True) -> str | None:
    """Commit sem pai com os ficheiros de `pasta` (na raiz) → ramo `ramo`. Devolve o hash, ou None se nada mudou."""
    if _git(repo, "branch", "--show-current") == ramo:
        raise RuntimeError(f"o ramo de trabalho é '{ramo}': o snapshot não pode substituir o ramo em uso")
    with tempfile.TemporaryDirectory() as tmp:        # índice à parte: o do repositório fica intacto
        env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
        _git(repo, f"--work-tree={pasta.resolve()}", "add", "--all", ".", env=env)
        tree = _git(repo, "write-tree", env=env)
    ref = f"refs/heads/{ramo}"
    atual = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", f"{ref}^{{tree}}"],
                           capture_output=True, text=True).stdout.strip()
    if atual == tree:
        return None
    commit = _git(repo, "commit-tree", tree, "-m", mensagem(asof))
    if push:
        _git(repo, "push", "--force", remoto, f"{commit}:{ref}")
    _git(repo, "update-ref", ref, commit)             # só depois do push: se falhar, tenta de novo no próximo run
    return commit[:7]


def main(forcar: bool = False) -> str | None:
    asof = json.loads((SNAPSHOT_LIVE / "meta.json").read_text())["dados_ate"]
    if not deve_publicar(asof, forcar=forcar):
        print(f"[publicador] {asof} não é sexta: snapshot não publicado (usar --publicar para forçar)")
        return None
    sha = publicar(ROOT, SNAPSHOT_LIVE, asof)
    print(f"[publicador] {'commit ' + sha + ' enviado para ' + SNAPSHOT_PUBLISH_REMOTE + '/' + SNAPSHOT_PUBLISH_BRANCH if sha else 'sem alterações'}")
    return sha


if __name__ == "__main__":
    main(forcar="--publicar" in sys.argv)
