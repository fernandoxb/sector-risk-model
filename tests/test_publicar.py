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


def test_commit_unico_no_ramo_snapshot_sem_tocar_no_main(repos, tmp_path):
    repo, remoto = repos
    fora = tmp_path / "dados"                                        # o snapshot vive fora do Git
    fora.mkdir()
    (fora / "meta.json").write_text('{"dados_ate": "2026-11-13"}')
    (repo / "codigo.py").write_text("x = 2\n")                      # alteração em curso: não pode ir
    git(repo, "add", "codigo.py")
    antes = git(repo, "rev-parse", "HEAD")
    sha = pb.publicar(repo, fora, "2026-11-13", ramo="snapshot")
    assert sha and git(remoto, "log", "-1", "--format=%s", "snapshot") == "Snapshot do cockpit: dados até 2026-11-13"
    assert git(remoto, "ls-tree", "--name-only", "snapshot") == "meta.json"          # só o snapshot, na raiz
    assert git(remoto, "rev-list", "--count", "snapshot") == "1"                     # commit sem pai
    assert git(repo, "rev-parse", "HEAD") == antes and git(remoto, "rev-parse", "main") == antes
    assert git(repo, "diff", "--cached", "--name-only") == "codigo.py"               # índice intacto
    assert pb.publicar(repo, fora, "2026-11-13", ramo="snapshot") is None            # nada mudou: sem commit
    (fora / "meta.json").write_text('{"dados_ate": "2026-11-20"}')
    (fora / "novo.csv").write_text("a\n1\n")
    assert pb.publicar(repo, fora, "2026-11-20", ramo="snapshot")
    assert git(remoto, "rev-list", "--count", "snapshot") == "1"                     # substitui, não acumula
    assert git(remoto, "ls-tree", "--name-only", "snapshot").split() == ["meta.json", "novo.csv"]


def test_recusa_substituir_o_ramo_de_trabalho(repos, tmp_path):
    repo, _ = repos
    with pytest.raises(RuntimeError, match="ramo em uso"):
        pb.publicar(repo, repo / "snap", "2026-11-13", ramo="main")


def test_push_falhado_volta_a_tentar(repos, tmp_path):
    repo, _ = repos
    fora = tmp_path / "dados"
    fora.mkdir()
    (fora / "meta.json").write_text("{}")
    with pytest.raises(RuntimeError):
        pb.publicar(repo, fora, "2026-11-13", ramo="snapshot", remoto="nao-existe")
    assert pb.publicar(repo, fora, "2026-11-13", ramo="snapshot")                    # o ramo local não ficou adiantado
