"""Strategies de treino: o contrato, o registry, e a estrategia neutra.

``supervised`` nao pertence a nenhuma trilha — e o piso da Parte 0 e o modelo que
entregamos. As estrategias **de trilha** moram em ``src/tracks/<trilha>/`` e se
registram quando ``src.tracks`` descobre a pasta, importado no fim deste arquivo.
Assim apagar uma trilha e apagar a pasta dela: nao ha import dela aqui.
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
from src import tracks  # noqa: E402,F401  isort:skip
