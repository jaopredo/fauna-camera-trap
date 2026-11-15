"""O par de vistas fraca/forte do FixMatch — aumentacao da trilha B.

A aula 13 (``DL_20242_13_Unsupervised.pdf``, p. 89) descreve o par: para cada imagem
sem rotulo, guarda-se a versao **fortemente** aumentada e calcula-se o pseudo-rotulo
sobre a **fracamente** aumentada. O par precisa vir da MESMA imagem, por isso uma
estrategia so devolve as duas vistas, em vez de duas estrategias independentes.

A vista fraca em si nao esta aqui: ela e ``src/data/transforms.py:build_weak``, porque
o modelo entregue tambem a usa e ela nao pertence a esta trilha.
"""

from __future__ import annotations

from typing import Any

from torchvision import transforms

from src.data.augment import AugmentationStrategy, register_augmentation
from src.data.transforms import DEFAULT_IMAGE_SIZE, to_tensor_ops, weak_pipeline_ops

#: Parametros do RandAugment na vista forte, como no paper.
STRONG_NUM_OPS = 2
STRONG_MAGNITUDE = 10

#: Fracao da imagem apagada pelo Cutout do paper, aqui via RandomErasing.
CUTOUT_SCALE = (0.02, 0.15)


class WeakStrongPair:
    """Devolve ``(fraca, forte)`` da mesma imagem.

    Classe em vez de closure porque o ``DataLoader`` com ``num_workers > 0``
    precisa serializar a transformacao para os processos filhos, e funcao aninhada
    nao e serializavel por ``pickle``.

    A ordem do ramo forte nao e arbitraria: ``RandAugment`` opera sobre PIL, e
    ``RandomErasing`` (o Cutout do paper) sobre tensor ja normalizado.
    """

    def __init__(
        self,
        *,
        image_size: int = DEFAULT_IMAGE_SIZE,
        horizontal_flip: float = 0.5,
        num_ops: int = STRONG_NUM_OPS,
        magnitude: int = STRONG_MAGNITUDE,
        cutout: float = 1.0,
    ) -> None:
        self.weak = transforms.Compose(
            [*weak_pipeline_ops(image_size, horizontal_flip), *to_tensor_ops()]
        )
        self.strong = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=horizontal_flip),
                transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),
                *to_tensor_ops(),
                transforms.RandomErasing(p=cutout, scale=CUTOUT_SCALE, value=0.0),
            ]
        )

    def __call__(self, image: Any) -> tuple[Any, Any]:
        return self.weak(image), self.strong(image)


@register_augmentation("weak_strong")
def build_weak_strong(
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    horizontal_flip: float = 0.5,
    num_ops: int = STRONG_NUM_OPS,
    magnitude: int = STRONG_MAGNITUDE,
    cutout: float = 1.0,
    **_: object,
) -> AugmentationStrategy:
    """O par de vistas do FixMatch, sobre a mesma imagem."""
    return WeakStrongPair(
        image_size=image_size,
        horizontal_flip=horizontal_flip,
        num_ops=num_ops,
        magnitude=magnitude,
        cutout=cutout,
    )
