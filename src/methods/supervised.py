"""Treino supervisionado puro — a estrategia do piso da Parte 0.

Consome apenas a fonte rotulada (os 105 exemplos do orcamento de 5) e aplica
entropia cruzada. E a implementacao de referencia do contrato ``TrainingStrategy``:
tudo o que as trilhas acrescentam — pseudo-rotulos, par fraco/forte, professor — e
medido contra o numero que sai daqui.

Sem peso por classe: ``rotulos_5.csv`` tem exatamente 5 imagens de cada uma das 21
classes, entao o conjunto de treino e balanceado por construcao. O desbalanceamento
do problema esta no ``val`` e no ``test_public``, e e por isso que a metrica
principal e o F1 macro, nao a perda.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import nn

from src.data.loaders import LABELED
from src.methods.base import Batch, StepOutput, TrainingStrategy
from src.methods.registry import register_strategy

if TYPE_CHECKING:
    from torch import nn as _nn


class SupervisedStrategy(TrainingStrategy):
    """Entropia cruzada sobre o conjunto rotulado."""

    required_sources = (LABELED,)
    primary_source = LABELED

    def __init__(self, *, label_smoothing: float = 0.0) -> None:
        self._loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def training_step(
        self, model: "_nn.Module", batch: Batch, *, device: Any
    ) -> StepOutput:
        images, labels = batch[LABELED]
        logits = model(images.to(device))
        loss = self._loss(logits, labels.to(device))
        return StepOutput(loss=loss, components={"ce": float(loss.detach())})


@register_strategy("supervised")
def build_supervised(*, label_smoothing: float = 0.0, **_: object) -> TrainingStrategy:
    """Constroi a estrategia supervisionada.

    ``label_smoothing`` fica em zero por padrao: a Parte 0 e o piso, e suavizar
    rotulo ja seria uma regularizacao a mais entre ele e o resultado das trilhas.
    """
    return SupervisedStrategy(label_smoothing=label_smoothing)
