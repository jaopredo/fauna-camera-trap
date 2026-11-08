"""Rastreamento de execucoes: seed, commit e manifesto."""

from __future__ import annotations

import json

import pytest

from src.tracking import (
    MANIFEST_NAME,
    create_run,
    git_commit,
    seed_everything,
    write_metrics,
)


def test_execucao_grava_manifesto_com_seed_e_commit(tmp_path):
    contexto = create_run(
        tmp_path, name="exp_b5_s0", seed=7, config={"name": "exp", "seed": 7}
    )
    assert contexto.run_dir == tmp_path / "exp_b5_s0"
    assert contexto.seed == 7

    manifesto = json.loads((contexto.run_dir / MANIFEST_NAME).read_text())
    assert manifesto["run"]["seed"] == 7
    assert manifesto["run"]["commit"]
    assert manifesto["config"]["name"] == "exp"
    assert manifesto["run"]["python"].startswith("3.")


def test_execucao_nao_apaga_artefato_anterior(tmp_path):
    """Retomar um treino depende do last.pt da execucao anterior."""
    primeira = create_run(tmp_path, name="run", seed=0)
    (primeira.run_dir / "last.pt").write_bytes(b"pesos")
    segunda = create_run(tmp_path, name="run", seed=0)
    assert (segunda.run_dir / "last.pt").read_bytes() == b"pesos"


def test_commit_do_repositorio_e_legivel(project_root):
    commit, _ = git_commit(project_root)
    assert commit != "unknown"
    assert len(commit) >= 7


def test_fora_de_repositorio_o_commit_e_desconhecido(tmp_path):
    commit, dirty = git_commit(tmp_path)
    assert (commit, dirty) == ("unknown", True)


def test_seed_torna_a_amostragem_repetivel():
    import torch

    seed_everything(123)
    primeiro = torch.randn(5)
    seed_everything(123)
    assert torch.allclose(primeiro, torch.randn(5))


def test_metricas_sao_gravadas_em_json(tmp_path):
    caminho = write_metrics(tmp_path, "metrics_val", {"macro_f1": 0.5})
    assert json.loads(caminho.read_text())["macro_f1"] == pytest.approx(0.5)
