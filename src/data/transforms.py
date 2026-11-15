"""Estrategias de aumentacao concretas, registradas por nome.

As aumentacoes que nao pertencem a nenhuma trilha:

``eval``         deterministica, usada na validacao, no teste e no ``predict.py``
``basic``        ``eval`` mais espelhamento horizontal (piso da Parte 0)
``weak``         a aumentacao fraca do FixMatch: espelhamento + translacao

**Por que redimensionar para um quadrado em vez de recortar.** Em um quarto das
imagens o animal ocupa menos de 2% do quadro e pode estar em qualquer canto; um
``CenterCrop`` o descartaria em boa parte dos casos. As proporcoes variam de camera
para camera porque a faixa de texto foi cortada do original, entao nao ha uma
proporcao comum a preservar. Distorcer a imagem e o preco; perder o animal, nao.

As aumentacoes especificas de uma trilha moram na pasta dela
(``src/tracks/b/transforms.py`` tem o par fraco/forte do FixMatch).

**Por que so espelhamento no treino.** O item 3 da Parte 0 pergunta em que epoca a
rede comeca a decorar as 105 imagens. Aumentacao forte adia a memorizacao e embaralha
exatamente a leitura que o enunciado pede. O espelhamento e seguro aqui: nenhuma das
21 especies depende de lateralidade para ser reconhecida. Aumentacao mais agressiva e
assunto do experimento obrigatorio da Trilha A.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torchvision import transforms

if (
    TYPE_CHECKING
):  # so para as anotacoes: este modulo nao precisa de numpy em tempo de execucao
    import numpy as np
    import torch
    from numpy.typing import NDArray

from src.data.augment import AugmentationStrategy, register_augmentation

#: Media e desvio dos canais RGB do ImageNet. Usados mesmo na CNN do zero para que
#: as duas versoes do piso vejam exatamente a mesma entrada e a comparacao isole a
#: arquitetura e os pesos herdados.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE = 224


def to_tensor_ops() -> list[Any]:
    """Tensor e normalizacao do ImageNet. Publico: as trilhas reusam."""
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
        [transforms.Resize((image_size, image_size)), *to_tensor_ops()]
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
            *to_tensor_ops(),
        ]
    )


# ===================== aumentacao fraca =====================================
# Espelhamento mais translacao. Nasceu como a vista fraca do FixMatch (aula 13,
# p. 88), mas **nao e da trilha B**: o modelo entregue (`p0_finetune_weak`) usa esta
# aumentacao, e ela e o controle do confundidor da Parte 1. Por isso fica aqui, na
# infraestrutura, e nao em `src/tracks/b/`. Apagar a trilha B nao pode levar embora
# a aumentacao do modelo que entregamos.

#: Deslocamento maximo, como fracao do lado (o FixMatch usa 12,5%).
WEAK_TRANSLATE = 0.125


def weak_pipeline_ops(image_size: int, horizontal_flip: float) -> list[Any]:
    """Espelhamento e translacao — perturbacao leve, que preserva a especie."""
    return [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=horizontal_flip),
        transforms.RandomAffine(degrees=0, translate=(WEAK_TRANSLATE, WEAK_TRANSLATE)),
    ]


@register_augmentation("weak")
def build_weak(
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    horizontal_flip: float = 0.5,
    **_: object,
) -> AugmentationStrategy:
    """Vista fraca do FixMatch, usada tambem no ramo rotulado (aula p. 88)."""
    return transforms.Compose(
        [*weak_pipeline_ops(image_size, horizontal_flip), *to_tensor_ops()]
    )


def as_displayable(tensor: "torch.Tensor") -> "NDArray[np.uint8]":
    """Desfaz ``ToTensor`` e ``Normalize``: devolve, em pixels, o que a rede recebeu.

    Serve as figuras da Parte 3. A alternativa seria redimensionar a imagem original
    de novo para desenha-la, e ai a figura e o modelo poderiam divergir — basta um
    metodo de interpolacao diferente entre o ``Resize`` do torchvision e o do PIL. Aqui
    a imagem desenhada e **literalmente** o tensor que entrou no modelo, invertido.

    O ``clamp`` e necessario porque a aumentacao pode produzir valor fora de 0-1 e o
    ``uint8`` daria a volta, pintando um pixel claro de escuro.
    """
    import numpy as np
    import torch

    media = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    desvio = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    pixels = (tensor.detach().cpu() * desvio + media).clamp(0, 1) * 255
    return pixels.round().byte().permute(1, 2, 0).numpy().astype(np.uint8)
