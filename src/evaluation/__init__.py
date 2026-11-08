"""Avaliacao e selecao de modelo."""

from src.evaluation.evaluator import EvaluationResult, evaluate, predict_labels
from src.evaluation.selection import (
    best_candidate,
    best_epoch,
    is_improvement,
    select_by,
)

__all__ = [
    "EvaluationResult",
    "best_candidate",
    "best_epoch",
    "evaluate",
    "is_improvement",
    "predict_labels",
    "select_by",
]
