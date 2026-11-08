"""Laco de treino e observadores."""

from src.training.loop import Trainer, TrainerError, build_optimizer
from src.training.observers import (
    CheckpointWriter,
    ConsoleLogger,
    EpochRecord,
    HistoryWriter,
    TrainingContext,
    TrainingObserver,
)

__all__ = [
    "CheckpointWriter",
    "ConsoleLogger",
    "EpochRecord",
    "HistoryWriter",
    "Trainer",
    "TrainerError",
    "TrainingContext",
    "TrainingObserver",
    "build_optimizer",
]
