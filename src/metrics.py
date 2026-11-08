"""Metricas de classificacao, como funcoes puras de ``(y_true, y_pred)``.

Modulo folha: depende so de numpy, nao conhece modelo, loader nem configuracao.
E consumido pelo treino (metrica de validacao por epoca) e pela avaliacao.

A armadilha que justifica isolar isto: **o F1 macro e sempre sobre as 21 classes**.
O ``val`` tem 202 imagens e 2 de cavalo; o ``test_public`` tem 4. Se uma classe nao
aparece nem no gabarito nem na predicao, uma media sobre "as classes observadas"
divide por 20 em vez de 21 e devolve um numero inflado, que nao se compara com
nenhum outro. Aqui o denominador e sempre ``num_classes``, e classe sem suporte
entra na media como F1 zero — a mesma convencao de
``sklearn.metrics.f1_score(labels=range(21), zero_division=0)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.errors import AppError


class MetricsError(AppError, ValueError):
    """Entrada incompativel com o numero de classes declarado."""


def _as_labels(values: ArrayLike, num_classes: int, name: str) -> NDArray[np.int64]:
    array = np.asarray(values, dtype=np.int64).reshape(-1)
    if array.size and (array.min() < 0 or array.max() >= num_classes):
        raise MetricsError(
            f"{name} tem indice fora da faixa 0..{num_classes - 1}: "
            f"min={array.min()}, max={array.max()}"
        )
    return array


def confusion_matrix(
    y_true: ArrayLike, y_pred: ArrayLike, *, num_classes: int
) -> NDArray[np.int64]:
    """Matriz ``num_classes x num_classes``; linha = verdadeiro, coluna = previsto."""
    true = _as_labels(y_true, num_classes, "y_true")
    pred = _as_labels(y_pred, num_classes, "y_pred")
    if true.size != pred.size:
        raise MetricsError(
            f"y_true e y_pred com tamanhos diferentes: {true.size} != {pred.size}"
        )
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def per_class_f1(confusion: NDArray[np.int64]) -> NDArray[np.float64]:
    """F1 de cada classe a partir da matriz de confusao.

    Uma classe sem nenhum acerto, sem predicao e sem suporte recebe F1 zero em vez
    de ``nan``: ela precisa pesar na media macro, nao desaparecer dela.
    """
    true_positive = np.diag(confusion).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    actual = confusion.sum(axis=1).astype(np.float64)
    denominator = predicted + actual
    return np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )


def macro_f1(confusion: NDArray[np.int64]) -> float:
    """Media simples do F1 por classe sobre **todas** as classes da matriz.

    Metrica principal do PA: pesa cavalo (4 imagens no test_public) igual a urso (82).
    """
    return float(per_class_f1(confusion).mean())


def accuracy(confusion: NDArray[np.int64]) -> float:
    total = int(confusion.sum())
    if total == 0:
        return 0.0
    return float(np.trace(confusion)) / total


@dataclass(frozen=True)
class ClassificationReport:
    """Numeros de uma avaliacao. Imutavel para poder ser guardado e comparado."""

    macro_f1: float
    accuracy: float
    per_class_f1: tuple[float, ...]
    support: tuple[int, ...]
    confusion: tuple[tuple[int, ...], ...]

    @property
    def num_classes(self) -> int:
        return len(self.per_class_f1)

    def as_dict(self) -> dict[str, object]:
        """Forma serializavel, para o manifesto da execucao."""
        return {
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "per_class_f1": list(self.per_class_f1),
            "support": list(self.support),
        }


def classification_report(
    y_true: ArrayLike, y_pred: ArrayLike, *, num_classes: int
) -> ClassificationReport:
    """Relatorio completo a partir dos rotulos verdadeiros e previstos."""
    confusion = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    return ClassificationReport(
        macro_f1=macro_f1(confusion),
        accuracy=accuracy(confusion),
        per_class_f1=tuple(float(v) for v in per_class_f1(confusion)),
        support=tuple(int(v) for v in confusion.sum(axis=1)),
        confusion=tuple(tuple(int(v) for v in row) for row in confusion),
    )
