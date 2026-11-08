"""Factory com Registry de Strategies de treino, por ``strategy.name`` no YAML.

Nenhuma estrategia concreta esta registrada ainda: isso depende da trilha
escolhida. Quando houver, o modulo e importado em ``src/methods/__init__.py``
para que o decorador rode.
"""

from __future__ import annotations

from typing import Any, Callable

from src.config.schema import StrategyConfig
from src.errors import AppError
from src.methods.base import TrainingStrategy

StrategyFactory = Callable[..., TrainingStrategy]

_REGISTRY: dict[str, StrategyFactory] = {}


class UnknownStrategyError(AppError, ValueError):
    """Nome de estrategia de treino que ninguem registrou."""


def register_strategy(name: str) -> Callable[[StrategyFactory], StrategyFactory]:
    def decorator(builder: StrategyFactory) -> StrategyFactory:
        if name in _REGISTRY:
            raise ValueError(f"estrategia ja registrada: {name!r}")
        _REGISTRY[name] = builder
        return builder

    return decorator


def available_strategies() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_strategy(cfg: StrategyConfig, **extra: Any) -> TrainingStrategy:
    """Instancia a estrategia declarada em ``cfg.name`` com os parametros do YAML."""
    try:
        builder = _REGISTRY[cfg.name]
    except KeyError:
        raise UnknownStrategyError(
            f"strategy.name desconhecido: {cfg.name!r}. "
            f"Disponiveis: {list(available_strategies())}"
        ) from None
    strategy = builder(**{**dict(cfg.params), **extra})
    if not isinstance(strategy, TrainingStrategy):
        raise TypeError(
            f"{cfg.name!r} devolveu {type(strategy).__name__}, que nao e um TrainingStrategy"
        )
    return strategy
