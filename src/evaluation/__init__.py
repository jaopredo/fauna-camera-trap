"""Avaliacao e selecao de modelo.

``src.evaluation.runs`` **nao** e exportado aqui de proposito. O que este pacote
expoe e a API de selecao, protegida pela barreira do ``test_public``
(``src.evaluation.selection``, com teste de contrato por introspecao). Reabrir uma
execucao para reportar numeros e outra coisa, e pode ver o ``test_public`` — quem
precisa importa ``from src.evaluation.runs import ...`` explicitamente, para que a
diferenca fique visivel na linha do import.
"""

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
