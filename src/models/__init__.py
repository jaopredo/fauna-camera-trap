"""Backbones, teto de parametros e checkpoints.

Os modulos com backbones concretos sao importados no fim deste arquivo para que o
decorador ``@register_model`` rode no import do pacote. Um backbone novo entra
acrescentando um modulo e uma linha de import ali.
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

# Importados pelo efeito colateral do decorador. Ficam no fim para nao criar ciclo
# com ``src.models.registry``, que estes modulos importam.
from src.models import resnet, small_cnn  # noqa: E402,F401  isort:skip
