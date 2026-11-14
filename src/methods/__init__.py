"""Strategies de treino. Cada trilha do enunciado entra como uma implementacao aqui.

Os modulos concretos sao importados no fim deste arquivo para que o decorador
``@register_strategy`` rode no import do pacote.
"""

from src.methods.base import Batch, StepOutput, TrainingStrategy
from src.methods.registry import (
    UnknownStrategyError,
    available_strategies,
    build_strategy,
    register_strategy,
)

__all__ = [
    "Batch",
    "StepOutput",
    "TrainingStrategy",
    "UnknownStrategyError",
    "available_strategies",
    "build_strategy",
    "register_strategy",
]

from src.methods import supervised  # noqa: E402,F401  isort:skip
