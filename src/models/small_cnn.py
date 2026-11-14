"""CNN pequena treinada do zero — versao (a) do piso da Parte 0.

O papel dela no trabalho e ser o chao: 105 imagens rotuladas, nenhum peso herdado.
O numero que ela produzir mede quanto do resultado vem da tarefa em si, e nao do
ImageNet nem dos dados sem rotulo.

Projeto deliberadamente simples e pequeno (~395 mil parametros, 3% do teto de 12 M):
quatro blocos convolucionais que dobram a largura e reduzem a resolucao pela metade,
seguidos de pooling global. Nao ha busca de arquitetura aqui — uma rede maior apenas
decoraria as 105 imagens mais depressa.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.registry import register_model

#: Larguras dos quatro blocos. Dobrar a cada bloco e a convencao mais comum e
#: mantem a contagem de parametros dominada pelo ultimo bloco.
WIDTHS = (32, 64, 128, 256)


class ConvBlock(nn.Module):
    """Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2.

    ``bias=False`` na convolucao porque a BatchNorm logo em seguida tem o proprio
    deslocamento: o vies da conv seria redundante e so gastaria parametro.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallCNN(nn.Module):
    """Classificador convolucional pequeno, treinado do zero.

    O pooling global antes da camada linear e o que torna a rede independente do
    tamanho da entrada e mantem a cabeca com poucos parametros — com 105 imagens,
    uma camada densa sobre o mapa achatado dominaria a contagem e decoraria ainda
    mais rapido.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        in_channels: int = 3,
        widths: tuple[int, ...] = WIDTHS,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        canais = (in_channels, *widths)
        self.features = nn.Sequential(
            *(ConvBlock(canais[i], canais[i + 1]) for i in range(len(widths)))
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(widths[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


@register_model("small_cnn")
def build_small_cnn(
    *,
    num_classes: int,
    pretrained: bool = False,
    widths: tuple[int, ...] | list[int] = WIDTHS,
    dropout: float = 0.5,
    **_: object,
) -> nn.Module:
    """Constroi a CNN pequena.

    ``pretrained`` existe no contrato do registry, mas aqui so pode ser ``False``:
    a versao (a) do enunciado e explicitamente "treinada do zero". Pedir pesos
    herdados para ela e erro de configuracao, nao uma variante silenciosa.
    """
    if pretrained:
        raise ValueError(
            "small_cnn nao tem pesos pre-treinados: a versao (a) da Parte 0 e "
            "treinada do zero. Use model.pretrained: false."
        )
    return SmallCNN(num_classes, widths=tuple(widths), dropout=dropout)
