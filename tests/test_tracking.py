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


# ===================== leitura de manifesto ==================================


def test_manifesto_e_lido_de_volta(tmp_path):
    from src.tracking import manifest_seed, read_manifest

    create_run(tmp_path, name="run", seed=42, config={"name": "x", "seeds": [0, 42]})
    manifesto = read_manifest(tmp_path / "run")
    assert manifest_seed(manifesto) == 42
    assert manifesto["config"]["seeds"] == [0, 42]


def test_manifesto_antigo_com_seed_singular_e_traduzido(tmp_path, monkeypatch):
    """Execucoes gravadas antes de 13/11 guardavam `seed` no bloco de config.

    A seed efetiva sempre esteve em `run.seed`, entao a traducao nao muda numero
    nenhum — e sem ela o relatorio nao conseguiria reler 38 min de treino.
    """
    import json

    from src.tracking import MANIFEST_NAME, manifest_seed, read_manifest

    run_dir = tmp_path / "antiga"
    run_dir.mkdir()
    (run_dir / MANIFEST_NAME).write_text(
        json.dumps({"run": {"seed": 7}, "config": {"name": "x", "seed": 7}}),
        encoding="utf-8",
    )

    manifesto = read_manifest(run_dir)
    assert manifesto["config"]["seeds"] == [7]
    assert "seed" not in manifesto["config"]
    assert manifest_seed(manifesto) == 7
    # o arquivo em disco nao e reescrito: o registro do experimento fica intacto
    bruto = json.loads((run_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert bruto["config"]["seed"] == 7


def test_manifesto_ausente_tem_erro_claro(tmp_path):
    import pytest as _pytest

    from src.tracking import read_manifest

    with _pytest.raises(FileNotFoundError, match="manifesto nao encontrado"):
        read_manifest(tmp_path / "nao_existe")


# ===================== escolha de dispositivo ================================


def test_dispositivo_explicito_e_respeitado():
    """`cpu` explicito continua valendo: e o que o predict.py usa."""
    from src.tracking import resolve_device

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_auto_escolhe_gpu_quando_existe(monkeypatch):
    """O padrao e auto para evitar o modo de falha caro: horas de treino em CPU
    numa maquina que tem GPU, sem nada no log avisando."""
    import torch

    from src.tracking import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_dispositivo_entra_no_manifesto(tmp_path):
    """Numero de GPU e de CPU so sao comparaveis entre si."""
    contexto = create_run(tmp_path, name="run", seed=0, device="cuda")
    manifesto = json.loads((contexto.run_dir / MANIFEST_NAME).read_text())
    assert manifesto["run"]["device"] == "cuda"


def test_manifesto_sem_dispositivo_assume_cpu(tmp_path):
    contexto = create_run(tmp_path, name="run", seed=0)
    assert contexto.device == "cpu"


def test_descricao_do_dispositivo_e_uma_linha():
    from src.tracking import describe_device

    assert "cpu" in describe_device("cpu")
