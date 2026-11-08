"""Reprodutibilidade de uma execucao: seed, commit, diretorio e manifesto.

O enunciado exige que toda tabela, curva e imagem da apresentacao seja reproduzivel
a partir do repositorio. Para isso cada execucao grava num diretorio proprio: a
configuracao usada, a seed, o commit do codigo e as metricas.

Modulo folha, consumido pelo treino, pela avaliacao e pela analise.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class RunContext:
    """Identidade de uma execucao."""

    run_dir: Path
    seed: int
    commit: str
    dirty: bool
    started_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "seed": self.seed,
            "commit": self.commit,
            "dirty": self.dirty,
            "started_at": self.started_at,
            "python": platform.python_version(),
        }


def git_commit(repo: Path | None = None) -> tuple[str, bool]:
    """Commit atual e se a arvore tem alteracao nao commitada.

    Devolve ``("unknown", True)`` fora de um repositorio: a execucao continua, mas
    o manifesto registra que o codigo nao e rastreavel.
    """
    cwd = str(repo or Path.cwd())
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return commit, bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Fixa as seeds de ``random``, numpy e torch.

    A seed sozinha nao basta em GPU: com ``cudnn.benchmark`` ligado, o cuDNN
    escolhe o algoritmo de convolucao medindo tempo, e a escolha pode mudar entre
    execucoes. As duas flags fixam algoritmos deterministicos. Isso garante
    repetir o resultado **na mesma maquina**: GPU, driver ou versao do PyTorch
    diferentes ainda dao numeros diferentes.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def create_run(
    output_dir: Path | str,
    *,
    name: str,
    seed: int,
    config: Mapping[str, Any] | None = None,
    repo: Path | None = None,
) -> RunContext:
    """Cria ``output_dir/name`` e grava o manifesto da execucao.

    Nao apaga o que ja existe: retomar um treino precisa do ``last.pt`` anterior.
    O manifesto e sobrescrito, registrando o commit da execucao mais recente.
    """
    run_dir = Path(output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = git_commit(repo)
    context = RunContext(
        run_dir=run_dir,
        seed=seed,
        commit=commit,
        dirty=dirty,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    manifest = {"run": context.as_dict(), "config": dict(config or {})}
    (run_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return context


def write_metrics(run_dir: Path, name: str, payload: Mapping[str, Any]) -> Path:
    """Grava um JSON de metricas no diretorio da execucao."""
    path = Path(run_dir) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path
