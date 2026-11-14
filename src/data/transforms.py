"""Estrategias de aumentacao concretas, registradas por nome.

Duas por enquanto, as que a Parte 0 precisa:

``eval``   deterministica, usada na validacao, no teste e no ``predict.py``
``basic``  ``eval`` mais espelhamento horizontal, usada no treino supervisionado

**Por que redimensionar para um quadrado em vez de recortar.** Em um quarto das
imagens o animal ocupa menos de 2% do quadro e pode estar em qualquer canto; um
``CenterCrop`` o descartaria em boa parte dos casos. As proporcoes variam de camera
para camera porque a faixa de texto foi cortada do original, entao nao ha uma
proporcao comum a preservar. Distorcer a imagem e o preco; perder o animal, nao.

**Por que so espelhamento no treino.** O item 3 da Parte 0 pergunta em que epoca a
rede comeca a decorar as 105 imagens. Aumentacao forte adia a memorizacao e embaralha
exatamente a leitura que o enunciado pede. O espelhamento e seguro aqui: nenhuma das
21 especies depende de lateralidade para ser reconhecida. Aumentacao mais agressiva e
assunto do experimento obrigatorio da Trilha A.
"""

from __future__ import annotations

from typing import Any

from torchvision import transforms

from src.data.augment import AugmentationStrategy, register_augmentation

#: Media e desvio dos canais RGB do ImageNet. Usados mesmo na CNN do zero para que
#: as duas versoes do piso vejam exatamente a mesma entrada e a comparacao isole a
#: arquitetura e os pesos herdados.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE = 224


def _to_tensor() -> list[Any]:
    return [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


@register_augmentation("eval")
def build_eval(
    *, image_size: int = DEFAULT_IMAGE_SIZE, **_: object
) -> AugmentationStrategy:
    """Entrada deterministica: a mesma imagem sempre vira o mesmo tensor.

    Determinismo nao e detalhe: a selecao de epoca e feita comparando F1 de
    validacao entre epocas, e uma transformacao aleatoria na validacao faria a
    metrica variar por motivo que nao e o modelo.
    """
    return transforms.Compose(
        [transforms.Resize((image_size, image_size)), *_to_tensor()]
    )


@register_augmentation("basic")
def build_basic(
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    horizontal_flip: float = 0.5,
    **_: object,
) -> AugmentationStrategy:
    """Treino supervisionado: ``eval`` mais espelhamento horizontal."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=horizontal_flip),
            *_to_tensor(),
        ]
    )
