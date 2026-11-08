"""Strategies de treino. Cada trilha do enunciado entra como uma implementacao aqui."""

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
