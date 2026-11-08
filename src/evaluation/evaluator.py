"""Avaliacao de um modelo num conjunto qualquer.

Uma funcao serve ``val`` e ``test_public`` porque a computacao e identica nos dois.
O que muda e a permissao, e ela vive no ``Split`` que o resultado carrega: toda
funcao de selecao em ``src.evaluation.selection`` recusa um resultado que nao venha
do ``val``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import torch

from src.data.splits import Split
from src.metrics import ClassificationReport, classification_report

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EvaluationResult:
    """Resultado de uma avaliacao, com a procedencia embutida.

    ``split`` nao e decoracao: e o que permite a selecao de modelo recusar o
    ``test_public`` sem depender de quem chamou lembrar da regra.
    """

    split: Split
    report: ClassificationReport
    #: confianca (probabilidade da classe prevista) por imagem, na ordem do loader
    confidence: tuple[float, ...] = ()
    y_true: tuple[int, ...] = ()
    y_pred: tuple[int, ...] = ()

    @property
    def macro_f1(self) -> float:
        return self.report.macro_f1

    @property
    def accuracy(self) -> float:
        return self.report.accuracy

    def metric(self, name: str) -> float:
        """Le uma metrica por nome — e assim que ``optim.monitor`` e resolvido."""
        try:
            value = getattr(self.report, name)
        except AttributeError:
            raise KeyError(
                f"metrica desconhecida: {name!r}. Disponiveis: macro_f1, accuracy"
            ) from None
        if not isinstance(value, float):
            raise KeyError(f"{name!r} nao e uma metrica escalar")
        return value


@torch.no_grad()
def evaluate(
    model: "nn.Module",
    loader: "DataLoader",
    *,
    split: Split,
    num_classes: int,
    device: Any = "cpu",
) -> EvaluationResult:
    """Roda o modelo em ``loader`` e devolve metricas sobre ``num_classes`` classes.

    Espera batches ``(imagens, rotulos)``. O modelo precisa devolver logits de
    forma ``(batch, num_classes)``.
    """
    was_training = model.training
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    confidence: list[float] = []
    try:
        for images, labels in loader:
            logits = model(images.to(device))
            if logits.ndim != 2 or logits.shape[1] != num_classes:
                raise ValueError(
                    f"o modelo devolveu logits de forma {tuple(logits.shape)}; "
                    f"esperado (batch, {num_classes})"
                )
            probabilities = torch.softmax(logits.float(), dim=1)
            top_probability, predicted = probabilities.max(dim=1)
            y_pred.extend(int(v) for v in predicted.cpu())
            y_true.extend(int(v) for v in torch.as_tensor(labels).cpu())
            confidence.extend(float(v) for v in top_probability.cpu())
    finally:
        model.train(was_training)

    report = classification_report(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        num_classes=num_classes,
    )
    return EvaluationResult(
        split=split,
        report=report,
        confidence=tuple(confidence),
        y_true=tuple(y_true),
        y_pred=tuple(y_pred),
    )


def predict_labels(
    model: "nn.Module",
    images: Iterable[Any],
    *,
    device: Any = "cpu",
) -> tuple[list[int], list[float]]:
    """Classe prevista e confianca para um iteravel de batches sem rotulo.

    Usado pelo ``predict.py`` (test_private) e pelos pseudo-rotuladores das
    trilhas B e C.
    """
    was_training = model.training
    model.eval()
    predictions: list[int] = []
    confidence: list[float] = []
    try:
        with torch.no_grad():
            for batch in images:
                logits = model(batch.to(device))
                probabilities = torch.softmax(logits.float(), dim=1)
                top_probability, predicted = probabilities.max(dim=1)
                predictions.extend(int(v) for v in predicted.cpu())
                confidence.extend(float(v) for v in top_probability.cpu())
    finally:
        model.train(was_training)
    return predictions, confidence
