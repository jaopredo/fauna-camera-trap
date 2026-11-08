"""Configuracao de experimento: schema congelado e carregamento."""

from src.config.loader import load_config
from src.config.schema import (
    ConfigError,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimConfig,
    StrategyConfig,
)

__all__ = [
    "ConfigError",
    "DataConfig",
    "ExperimentConfig",
    "StrategyConfig",
    "ModelConfig",
    "OptimConfig",
    "load_config",
]
