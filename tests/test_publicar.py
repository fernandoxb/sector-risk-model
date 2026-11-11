"""Testes do bot Publicador com um repositório git temporário (sem rede: o remoto é uma pasta local)."""
import subprocess
from pathlib import Path

import pytest

from src.report import publicar as pb


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repos(tmp_path):
    remoto, repo = tmp_path / "remoto.git", tmp_path / "repo"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(remoto)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    git(repo, "config", "user.email", "teste@exemplo.invalid")
    git(repo, "config", "user.name", "Teste")
    (repo / "snap").mkdir()
    (repo / "snap" / "meta.json").write_text('{"dados_ate": "2026-11-06"}')
    (repo / "codigo.py").write_text("x = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "inicial")
    git(repo, "remote", "add", "origin", str(remoto))
    git(repo, "push", "origin", "main")
    return repo, remoto


def test_so_publica_as_sextas():
    assert pb.deve_publicar("2026-11-13") and not pb.deve_publicar("2026-11-12")
    assert pb.deve_publicar("2026-11-12", forcar=True) and not pb.deve_publicar("2026-11-13", ativo=False)


def test_commit_so_do_snapshot_e_push(repos):
    repo, remoto = repos
    (repo / "snap" / "meta.json").write_text('{"dados_ate": "2026-11-13"}')
    (repo / "codigo.py").write_text("x = 2\n")                      # alteração em curso: não pode ir
    git(repo, "add", "codigo.py")                                   # nem mesmo se já estiver no índice
    sha = pb.publicar(repo, repo / "snap", "2026-11-13")
    assert sha and git(remoto, "log", "-1", "--format=%s") == "Snapshot do cockpit: dados até 2026-11-13"
    assert git(repo, "show", "--name-only", "--format=", "HEAD") == "snap/meta.json"
    assert "codigo.py" in git(repo, "diff", "--cached", "--name-only")   # continua por publicar, intacto
    assert pb.publicar(repo, repo / "snap", "2026-11-13") is None        # nada mudou: sem commit


def test_recusa_outro_ramo(repos):
    repo, _ = repos
    git(repo, "checkout", "-b", "experiencia")
    (repo / "snap" / "meta.json").write_text("{}")
    with pytest.raises(RuntimeError, match="experiencia"):
        pb.publicar(repo, repo / "snap", "2026-11-13")
