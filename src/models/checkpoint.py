"""Gravacao e leitura de checkpoints.

Dois arquivos com papeis diferentes, de proposito:

``last.pt``  estado completo (pesos + otimizador + epoca + historico) para retomar
             um treino interrompido — sessao do Colab que expira, queda de energia,
             as varias seeds da Parte 2.
``best.pt``  o modelo escolhido no ``val``; e ele que vai para a avaliacao e para a
             entrega. Pode ser de uma epoca bem anterior a ultima.

Retomar so com os pesos nao basta: o otimizador guarda medias moveis dos gradientes,
e recomecar sem elas deixa as primeiras epocas depois do resume instaveis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from src.errors import AppError

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer

BEST_NAME = "best.pt"
LAST_NAME = "last.pt"


class CheckpointError(AppError, RuntimeError):
    """Checkpoint ausente ou incompativel com o modelo pedido."""


@dataclass(frozen=True)
class CheckpointState:
    """O que um checkpoint carrega, alem dos pesos."""

    epoch: int
    best_metric: float
    monitor: str
    history: tuple[Mapping[str, Any], ...] = ()
    config: Mapping[str, Any] | None = None
    #: estado proprio do metodo de treino (ex.: EMA do professor, limiar por
    #: classe). Separado de ``config``: um e o que foi pedido, o outro e o que o
    #: metodo aprendeu e precisa de volta para retomar sem descontinuidade.
    strategy_state: Mapping[str, Any] = field(default_factory=dict)


def save_checkpoint(
    path: Path,
    *,
    model: "nn.Module",
    state: CheckpointState,
    optimizer: "Optimizer | None" = None,
) -> Path:
    """Grava pesos e estado em ``path`` (criando o diretorio)."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": state.epoch,
        "best_metric": state.best_metric,
        "monitor": state.monitor,
        "history": list(state.history),
        "config": dict(state.config) if state.config else None,
        "strategy_state": dict(state.strategy_state),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Path,
    *,
    model: "nn.Module",
    optimizer: "Optimizer | None" = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> CheckpointState:
    """Carrega pesos (e opcionalmente o otimizador) de ``path``.

    ``map_location="cpu"`` por padrao porque o modelo entregue roda em CPU e os
    pesos podem ter sido salvos em GPU. ``weights_only=True`` recusa payload com
    objeto arbitrario — o checkpoint e um arquivo de dados, nao de codigo.
    """
    import torch

    if not path.exists():
        raise CheckpointError(f"checkpoint nao encontrado: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    missing, unexpected = model.load_state_dict(payload["model"], strict=strict)
    if missing or unexpected:
        raise CheckpointError(
            f"{path}: pesos incompativeis. Ausentes: {list(missing)[:5]}; "
            f"inesperados: {list(unexpected)[:5]}"
        )
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    history: Sequence[Mapping[str, Any]] = payload.get("history") or ()
    return CheckpointState(
        epoch=int(payload.get("epoch", 0)),
        best_metric=float(payload.get("best_metric", float("-inf"))),
        monitor=str(payload.get("monitor", "macro_f1")),
        history=tuple(history),
        config=payload.get("config"),
        strategy_state=payload.get("strategy_state") or {},
    )
