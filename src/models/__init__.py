"""Backbones, teto de parametros e checkpoints.

Os modulos com backbones concretos sao importados aqui para que o decorador
``@register_model`` rode no import do pacote. Nenhum existe ainda.
"""

from src.models.budget import (
    MAX_PARAMETERS,
    ParameterBudgetError,
    assert_within_budget,
    count_parameters,
)
from src.models.checkpoint import (
    BEST_NAME,
    LAST_NAME,
    CheckpointError,
    CheckpointState,
    load_checkpoint,
    save_checkpoint,
)
from src.models.registry import (
    UnknownModelError,
    available_models,
    build_model,
    register_model,
)

__all__ = [
    "BEST_NAME",
    "CheckpointError",
    "CheckpointState",
    "LAST_NAME",
    "MAX_PARAMETERS",
    "ParameterBudgetError",
    "UnknownModelError",
    "assert_within_budget",
    "available_models",
    "build_model",
    "count_parameters",
    "load_checkpoint",
    "register_model",
    "save_checkpoint",
]
