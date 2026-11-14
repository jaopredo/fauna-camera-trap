"""ResNet-18 com pesos de ImageNet — versao (b) do piso e backbone do resto do PA.

Por que esta arquitetura (ver ``docs/decisions.md``, D8): o teto de referencia do
enunciado (0,686 de F1 macro) foi medido com uma ResNet-18, entao a curva da Parte 2
compara maca com maca. O enunciado a nomeia entre as que cabem no teto de 12 M, e a
aula 13 (``DL_20242_13_Unsupervised.pdf``, p. 60) a usa como backbone.

Receita de fine-tuning da aula 15 (``DL_20242_15_VisualFM.pdf``, p. 15): remover a
cabeca pre-treinada e colocar uma camada nova com o numero de classes do problema,
depois treinar nos dados disponiveis.
"""

from __future__ import annotations

from torch import nn

from src.models.registry import register_model


@register_model("resnet18")
def build_resnet18(
    *,
    num_classes: int,
    pretrained: bool = False,
    **_: object,
) -> nn.Module:
    """ResNet-18 com a camada final trocada por uma de ``num_classes`` saidas.

    Com ``pretrained=True`` o torchvision **baixa** os pesos de ImageNet — isso vale
    no treino, onde ha internet. Na inferencia nao: o ``predict.py`` forca
    ``pretrained=False`` e carrega o checkpoint local, de modo que nenhum download
    acontece ao rodar o modelo entregue.

    A troca da ``fc`` tambem derruba a contagem de parametros: a cabeca de 1000
    classes do ImageNet tem 513.000 parametros, e a de 21 tem 10.773.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
