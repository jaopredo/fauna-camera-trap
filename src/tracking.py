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
    device: str = "cpu"

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "seed": self.seed,
            "commit": self.commit,
            "dirty": self.dirty,
            "started_at": self.started_at,
            "python": platform.python_version(),
            # o dispositivo entra no manifesto porque numeros de GPU e de CPU so
            # sao comparaveis entre si: ordem de soma e algoritmo de convolucao
            # mudam, e a diferenca aparece nas casas decimais da metrica
            "device": self.device,
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


def resolve_device(preference: str = "auto") -> str:
    """Traduz ``auto`` no melhor dispositivo disponivel.

    O padrao e ``auto`` justamente para evitar o modo de falha caro: treinar horas
    em CPU sem perceber, numa maquina que tem GPU. Com ROCm o dispositivo continua
    se chamando ``cuda`` — o PyTorch expoe a Radeon pela mesma API.

    ``cpu`` explicito continua valendo: e o que o ``predict.py`` usa, porque o
    enunciado exige que o modelo entregue rode em CPU.
    """
    if preference != "auto":
        return preference

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_device(device: str) -> str:
    """Uma linha dizendo onde a execucao vai rodar."""
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        propriedades = torch.cuda.get_device_properties(0)
        return (
            f"{device} | {propriedades.name} | {propriedades.total_memory / 1e9:.1f} GB"
        )
    return f"{device} | {platform.processor() or 'CPU'}"


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
    device: str = "cpu",
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
        device=device,
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


def read_manifest(run_dir: Path | str) -> dict[str, Any]:
    """Le o manifesto de uma execucao, normalizando formatos antigos.

    Execucoes gravadas antes de 13/11/2025 guardavam ``seed`` (singular) no bloco
    de configuracao, antes de o experimento passar a declarar ``seeds``. A seed
    efetiva da execucao sempre esteve em ``run.seed``, entao a traducao nao muda
    numero nenhum — so a forma — e o manifesto em disco fica intacto: o registro
    do experimento nao e reescrito.
    """
    path = Path(run_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"manifesto nao encontrado: {path}")
    manifesto = json.loads(path.read_text(encoding="utf-8"))

    config = dict(manifesto.get("config") or {})
    if "seed" in config and "seeds" not in config:
        config["seeds"] = [config.pop("seed")]
    manifesto["config"] = config
    return manifesto


def manifest_seed(manifesto: Mapping[str, Any]) -> int:
    """A seed **desta** execucao.

    Vem de ``run.seed``, nunca do bloco de configuracao: o config declara o
    conjunto de seeds do experimento, e o diretorio corresponde a uma delas.
    """
    return int(manifesto["run"]["seed"])
