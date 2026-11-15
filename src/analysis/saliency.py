"""Para onde o modelo olhou: CAM, medidas do mapa e teste de fidelidade por oclusao.

A Parte 3 do enunciado pede "um mapa de saliencia (Grad-CAM ou oclusao) que mostre
para onde o modelo olhou". Usamos **CAM** (Class Activation Mapping), de
``docs/DL_20242_13_Unsupervised.pdf``, p. 80-82, sob "Weakly-Supervised Learning"
(Zhou et al., CVPR 2016) — o ``CLAUDE.md`` manda preferir o que foi visto em aula, e
aqui a opcao da aula e tambem a mais simples.

**Por que isso satisfaz o "Grad-CAM" do enunciado.** Numa rede que termina em global
average pooling seguido de uma camada linear — e a ResNet-18 termina exatamente
assim, ``avgpool`` -> ``fc`` — os dois metodos dao o *mesmo mapa*, a menos de uma
constante positiva. Com ``A_k`` o k-esimo mapa de ativacao de ``layer4``, ``Z`` o
numero de posicoes e ``w_k^c`` o peso da ``fc``::

    y^c = sum_k w_k^c * (1/Z) * sum_ij A_k(i,j)
    => d y^c / d A_k(i,j) = w_k^c / Z
    => alpha_k^c = (1/Z) * sum_ij d y^c / d A_k(i,j) = w_k^c / Z

O peso do Grad-CAM e o peso da ``fc`` dividido por ``Z``, logo
``ReLU(sum_k alpha_k A_k) = (1/Z) * ReLU(sum_k w_k A_k)``. Nao e analogia: e
igualdade, e ``tests/test_saliency.py`` a verifica numericamente contra o autograd.
A consequencia pratica e que **nao ha nada a treinar e nenhum hook a instalar** — os
``w_k^c`` sao os pesos que o treino ja aprendeu.

**Limite de resolucao, que vale dizer na apresentacao.** Em 224x224 a ``layer4``
devolve uma grade 7x7: cada celula do mapa cobre 32x32 pixels da imagem. O
redimensionamento para 224 e interpolacao, nao informacao nova. Portanto o mapa
sustenta afirmacoes do tipo "a evidencia esta no canto inferior esquerdo, nao no
centro", e **nao** sustenta "o mapa esta exatamente na pata do animal".

Divisao de responsabilidade: aqui ficam o mapa e as medidas **do mapa**; a escolha de
quais imagens mostrar e o diagnostico ficam em ``src.analysis.gallery``; o desenho,
em ``src.analysis.figures``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.errors import AppError

#: Fracao dos pixels mais quentes que define "a regiao que o modelo usou". 10% de
#: 224x224 sao ~5.000 pixels, area equivalente a um quadrado de 71x71 — grande o
#: bastante para conter um animal de porte medio, pequena o bastante para que
#: conte-la como "a regiao atendida" signifique algo.
TOP_MASS_FRACTION = 0.10

#: Largura da faixa de borda, como fracao do lado. A faixa e o que sobra ao remover
#: o retangulo central de (1 - 2*0.2) = 60% do lado, logo ocupa 64% da area — por
#: isso a medida de borda e sempre reportada ao lado da area que a faixa ocupa, que e
#: a referencia nula de um mapa uniforme.
BORDER_FRACTION = 0.20

#: Nivel a partir do qual um pixel conta como "quente", na escala 0-1 do mapa
#: normalizado pelo proprio pico. Meia altura e a convencao usual para medir a
#: largura de um pico.
HOT_LEVEL = 0.5


class SaliencyError(AppError, ValueError):
    """Modelo incompativel com CAM, ou entradas de formas inconsistentes."""


# ===================== decompor o modelo ====================================


@dataclass(frozen=True)
class CamModel:
    """Um modelo partido nas duas pecas que o CAM precisa.

    ``features`` vai da imagem aos mapas de ativacao (saida da ``layer4``, antes do
    pooling); ``weight`` e a matriz da camada linear final, ``(num_classes, k)``.

    Nao ha copia de peso: os modulos e tensores sao os mesmos objetos do modelo
    original, so vistos de outro angulo. Mover o modelo para a GPU move isto junto.
    """

    features: nn.Module
    weight: Tensor
    bias: Tensor | None

    @property
    def num_classes(self) -> int:
        return int(self.weight.shape[0])

    def forward_features(self, images: Tensor) -> Tensor:
        """Mapas de ativacao ``(n, k, h, w)``."""
        return self.features(images)

    def logits_from_features(self, features: Tensor) -> Tensor:
        """Logits ``(n, num_classes)`` a partir dos mapas — o GAP mais a linear.

        Tem de coincidir com ``model(images)`` ate a precisao numerica, senao o mapa
        explicaria um modelo que nao e o avaliado. ``tests/test_saliency.py`` compara
        os dois na ResNet-18 de verdade.
        """
        pooled = features.mean(dim=(2, 3))
        return F.linear(pooled, self.weight, self.bias)


def decompose(model: nn.Module) -> CamModel:
    """Parte o modelo em ``features`` e cabeca linear, exigindo a forma que CAM pede.

    O requisito do CAM e estrutural: global average pooling seguido de **uma** camada
    linear. A verificacao e explicita e levanta excecao porque a alternativa e pior —
    num modelo de outra forma (a ``small_cnn`` do zero, por exemplo, se ela tivesse
    camadas densas no fim) a conta ainda produziria um mapa, apenas sem significado.
    Falhar alto e melhor que uma figura plausivel e errada.
    """
    filhos = list(model.children())
    if len(filhos) < 3:
        raise SaliencyError(
            f"modelo com {len(filhos)} submodulos; CAM precisa de pelo menos "
            f"features + pooling + linear"
        )
    cabeca = filhos[-1]
    if not isinstance(cabeca, nn.Linear):
        raise SaliencyError(
            f"a ultima camada e {type(cabeca).__name__}, nao nn.Linear; CAM exige "
            f"uma unica camada linear sobre o pooling"
        )
    # Flatten, Identity e Dropout entre o pooling e a linear nao mudam o mapa: em
    # modo de avaliacao os dois ultimos sao identidade, e o achatamento e o que esta
    # funcao faz por conta propria em `logits_from_features`. A ResNet-18 nao tem
    # nenhum deles (ela achata dentro do `forward`), mas um `nn.Sequential` escrito a
    # mao tem, e recusa-lo seria exigir uma forma de codigo em vez de uma de modelo.
    corpo = filhos[:-1]
    while corpo and isinstance(corpo[-1], (nn.Flatten, nn.Identity, nn.Dropout)):
        corpo = corpo[:-1]
    if not corpo:
        raise SaliencyError("modelo sem camadas antes da linear final")
    pooling = corpo[-1]
    if not isinstance(pooling, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
        raise SaliencyError(
            f"a camada antes da linear e {type(pooling).__name__}, nao um average "
            f"pooling; sem GAP os pesos da linear nao sao os w_k^c do CAM"
        )
    return CamModel(
        features=nn.Sequential(*corpo[:-1]),
        weight=cabeca.weight,
        bias=cabeca.bias,
    )


# ===================== o mapa ===============================================


def class_activation_maps(
    features: Tensor, weight: Tensor, class_indices: Sequence[int] | Tensor
) -> Tensor:
    """Mapa CAM bruto ``(n, h, w)``: a soma dos mapas de ativacao pesada pela ``fc``.

    Bruto significa **com sinal**: valor negativo e evidencia *contra* a classe, e
    quem normaliza e ``heatmaps``. Separar os dois passos deixa o teste comparar com
    o autograd antes de qualquer normalizacao.
    """
    if features.ndim != 4:
        raise SaliencyError(
            f"features deveria ser (n,k,h,w); veio {tuple(features.shape)}"
        )
    indices = torch.as_tensor(class_indices, dtype=torch.long, device=weight.device)
    if indices.ndim != 1 or indices.numel() != features.shape[0]:
        raise SaliencyError(
            f"{indices.numel()} classes para {features.shape[0]} imagens"
        )
    if int(indices.max()) >= weight.shape[0] or int(indices.min()) < 0:
        raise SaliencyError(f"indice de classe fora de 0..{weight.shape[0] - 1}")
    pesos = weight[indices].to(features.dtype)
    return torch.einsum("nk,nkhw->nhw", pesos, features)


def heatmaps(maps: Tensor, *, size: int) -> Tensor:
    """Mapas em ``[0,1]``, ampliados para ``size x size`` e normalizados pelo pico.

    Tres passos, nesta ordem e por estes motivos:

    1. **ReLU** — descarta a evidencia contraria, e e exatamente onde CAM e Grad-CAM
       coincidem (ver o cabecalho do modulo);
    2. **interpolacao bilinear** da grade 7x7 para a resolucao da imagem, para poder
       sobrepor ao que o modelo viu;
    3. **divisao pelo pico**, nao min-max: assim o zero continua significando "nenhuma
       evidencia positiva aqui", e as medidas de massa do mapa fazem sentido. Um
       min-max transformaria o menor valor em zero mesmo quando ele e alto.

    Mapa sem nenhum valor positivo (possivel ao pedir o mapa de uma classe que o
    modelo rejeita) devolve zeros, em vez de dividir por zero.
    """
    positivo = torch.relu(maps.float())
    if positivo.ndim != 3:
        raise SaliencyError(f"maps deveria ser (n,h,w); veio {tuple(maps.shape)}")
    ampliado = F.interpolate(
        positivo.unsqueeze(1), size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(1)
    picos = ampliado.amax(dim=(1, 2), keepdim=True)
    return torch.where(picos > 0, ampliado / picos, torch.zeros_like(ampliado))


# ===================== medidas do mapa ======================================


@dataclass(frozen=True)
class MapStatistics:
    """O que se mede de um mapa de calor, sem olhar a imagem.

    Nenhuma destas medidas depende de rotulo, de caixa do animal ou de alguem olhar a
    figura — e por isso que elas podem entrar num diagnostico reprodutivel. O que elas
    **nao** dizem e se o que esta sob o mapa e um animal: isso exige a caixa, que o
    pacote nao traz. Ver ``src.analysis.gallery``.
    """

    #: fracao da massa total contida nos pixels mais quentes (ver TOP_MASS_FRACTION)
    concentration: float
    #: fracao da massa que cai na faixa de borda
    border_mass: float
    #: fracao da AREA que a faixa de borda ocupa — a referencia nula
    border_area: float
    peak_row: int
    peak_col: int
    #: pixels com calor >= HOT_LEVEL
    hot_pixels: int
    #: lado do mapa, para ler hot_pixels em contexto
    side: int

    @property
    def hot_side(self) -> float:
        """Lado do quadrado de area equivalente a regiao quente, em pixels.

        E o numero que deixa o diagnostico falar a lingua do enunciado ("o animal
        ocupa 40x30 pixels depois do redimensionamento"): aqui nao e o animal que se
        mede, e a **regiao em que o modelo se baseou**.
        """
        return math.sqrt(self.hot_pixels)

    @property
    def border_excess(self) -> float:
        """Massa na borda menos a area da borda. Positivo = mapa puxado para a moldura."""
        return self.border_mass - self.border_area

    def as_row(self) -> dict[str, object]:
        return {
            "concentracao_top10": round(self.concentration, 4),
            "massa_na_borda": round(self.border_mass, 4),
            "area_da_borda": round(self.border_area, 4),
            "excesso_na_borda": round(self.border_excess, 4),
            "pico_linha": self.peak_row,
            "pico_coluna": self.peak_col,
            "pixels_quentes": self.hot_pixels,
            "lado_equivalente_px": round(self.hot_side, 1),
        }


def _as_numpy(heatmap: Tensor | NDArray[np.float64]) -> NDArray[np.float64]:
    if isinstance(heatmap, Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    mapa = np.asarray(heatmap, dtype=np.float64)
    if mapa.ndim != 2:
        raise SaliencyError(f"esperava um mapa 2-D; veio {mapa.shape}")
    return mapa


def _region(
    heatmap: Tensor | NDArray[np.float64], fraction: float, *, coldest: bool
) -> NDArray[np.bool_]:
    """Os ``fraction`` pixels mais quentes (ou mais frios) do mapa, como mascara.

    Seleciona por **quantidade de pixels**, nao por nivel de calor: assim a regiao
    quente e a fria tem exatamente a mesma area, e compara-las — ou ocluir uma e a
    outra — compara coisas do mesmo tamanho. Um corte por nivel daria areas diferentes
    e tornaria a comparacao sem sentido.

    **O desempate importa, e e a parte nao obvia.** Um mapa normalizado tem milhares
    de pixels exatamente iguais (zero fora da regiao de evidencia, ou 1,0 num mapa
    saturado), entao o corte quase sempre cai dentro de um empate grande. Resolver o
    empate pela ordem do array escolheria sempre as primeiras linhas da imagem, do
    canto superior esquerdo — e com isso:

    * a regiao **quente** ganharia pixels distantes do pico, deixando de ser uma regiao;
    * a regiao **fria**, que e o controle da oclusao, cairia na faixa de cima da foto,
      que em camera trap e quase sempre ceu ou folhagem. O controle sairia
      artificialmente facil de apagar e o teste de fidelidade, artificialmente
      generoso.

    Por isso o empate e resolvido pela distancia ao pico: a quente puxa para perto, a
    fria para longe. As duas saem como regioes coerentes, em lados opostos.
    """
    mapa = _as_numpy(heatmap)
    if not 0 < fraction <= 1:
        raise SaliencyError(f"fraction tem de estar em (0,1]; veio {fraction}")
    quantos = max(1, int(round(fraction * mapa.size)))
    pico = np.unravel_index(int(np.argmax(mapa)), mapa.shape)
    linhas, colunas = np.indices(mapa.shape)
    distancia = np.hypot(linhas - pico[0], colunas - pico[1]).ravel()
    # lexsort usa a ULTIMA chave como primaria, e ordena crescente
    primaria = mapa.ravel() if coldest else -mapa.ravel()
    secundaria = -distancia if coldest else distancia
    ordem = np.lexsort((secundaria, primaria))
    mascara = np.zeros(mapa.size, dtype=bool)
    mascara[ordem[:quantos]] = True
    return mascara.reshape(mapa.shape)


def top_mass_mask(
    heatmap: Tensor | NDArray[np.float64], *, fraction: float = TOP_MASS_FRACTION
) -> NDArray[np.bool_]:
    """Mascara dos pixels mais quentes — "a regiao em que o modelo se baseou"."""
    return _region(heatmap, fraction, coldest=False)


def coldest_mask(
    heatmap: Tensor | NDArray[np.float64], *, fraction: float = TOP_MASS_FRACTION
) -> NDArray[np.bool_]:
    """Mascara dos pixels mais frios, com a mesma area de ``top_mass_mask``.

    E o controle da oclusao: apagar 10% da imagem derruba a confianca de qualquer
    modelo um pouco. A pergunta que interessa nao e "apagar a regiao quente derruba?",
    e "derruba **mais** do que apagar uma regiao de area igual que o mapa ignorou?".
    """
    return _region(heatmap, fraction, coldest=True)


def mask_iou(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    """Intersecao sobre uniao de duas mascaras. 1 = mesma regiao, 0 = disjuntas."""
    primeira, segunda = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if primeira.shape != segunda.shape:
        raise SaliencyError(
            f"mascaras de formas diferentes: {primeira.shape} e {segunda.shape}"
        )
    uniao = int((primeira | segunda).sum())
    return float((primeira & segunda).sum() / uniao) if uniao else 0.0


def map_statistics(
    heatmap: Tensor | NDArray[np.float64],
    *,
    top_fraction: float = TOP_MASS_FRACTION,
    border_fraction: float = BORDER_FRACTION,
    hot_level: float = HOT_LEVEL,
) -> MapStatistics:
    """Mede concentracao, massa na borda, pico e tamanho da regiao quente."""
    mapa = _as_numpy(heatmap)
    lado = mapa.shape[0]
    massa = float(mapa.sum())

    quentes = top_mass_mask(mapa, fraction=top_fraction)
    concentracao = float(mapa[quentes].sum() / massa) if massa > 0 else 0.0

    faixa = max(1, int(round(border_fraction * lado)))
    centro = np.zeros_like(mapa, dtype=bool)
    centro[faixa : mapa.shape[0] - faixa, faixa : mapa.shape[1] - faixa] = True
    borda = ~centro
    massa_borda = float(mapa[borda].sum() / massa) if massa > 0 else 0.0

    pico = np.unravel_index(int(np.argmax(mapa)), mapa.shape)
    return MapStatistics(
        concentration=concentracao,
        border_mass=massa_borda,
        border_area=float(borda.sum() / mapa.size),
        peak_row=int(pico[0]),
        peak_col=int(pico[1]),
        hot_pixels=int((mapa >= hot_level).sum()),
        side=lado,
    )


# ===================== o mapa explica a decisao? ============================


@dataclass(frozen=True)
class Faithfulness:
    """A oclusao como **teste** do mapa, nao como segundo mapa.

    Um mapa de saliencia e uma afirmacao sobre a decisao do modelo, e afirmacao se
    testa. Apagamos a regiao que o mapa aponta e medimos o que acontece com a
    probabilidade da classe prevista. Tres medidas, porque duas nao bastam:

    * ``without_hot`` — apagando a regiao atendida;
    * ``without_cold`` — apagando uma regiao de **area igual** que o mapa ignorou. Sem
      este controle, qualquer queda seria confundida com "apagar 10% da imagem
      atrapalha";
    * ``only_hot`` — mantendo **so** a regiao atendida. Alto aqui significa que a
      regiao basta para a decisao.

    A regiao apagada e preenchida com zero **no espaco normalizado**, que depois da
    normalizacao do ImageNet corresponde ao pixel medio — o cinza que a rede ve como
    "nada de especial". Preencher com preto seria injetar um padrao de alto contraste.
    """

    baseline: float
    without_hot: float
    without_cold: float
    only_hot: float

    def drop_hot(self) -> float:
        return self.baseline - self.without_hot

    def drop_cold(self) -> float:
        return self.baseline - self.without_cold

    def explains(self) -> bool:
        """O mapa explica a decisao: apagar o que ele aponta custa mais que o controle."""
        return self.drop_hot() > self.drop_cold()

    def as_row(self) -> dict[str, object]:
        return {
            "confianca": round(self.baseline, 4),
            "sem_a_regiao_quente": round(self.without_hot, 4),
            "sem_a_regiao_fria": round(self.without_cold, 4),
            "so_a_regiao_quente": round(self.only_hot, 4),
            "queda_quente": round(self.drop_hot(), 4),
            "queda_fria": round(self.drop_cold(), 4),
            "o_mapa_explica": self.explains(),
        }


@torch.no_grad()
def faithfulness(
    model: nn.Module,
    images: Tensor,
    heats: Sequence[Tensor | NDArray[np.float64]],
    class_indices: Sequence[int],
    *,
    device: str,
    fraction: float = TOP_MASS_FRACTION,
) -> tuple[Faithfulness, ...]:
    """Testa, por imagem, se o mapa explica a decisao. Quatro forwards por imagem.

    ``images`` e o lote **ja normalizado**, como o modelo recebe; ``heats`` tem um mapa
    por imagem, na resolucao de ``images``.
    """
    if len(heats) != images.shape[0] or len(class_indices) != images.shape[0]:
        raise SaliencyError(
            f"{images.shape[0]} imagens, {len(heats)} mapas e "
            f"{len(class_indices)} classes"
        )
    model.eval()
    lote = images.to(device)
    quentes = torch.stack(
        [
            torch.as_tensor(top_mass_mask(h, fraction=fraction), device=device)
            for h in heats
        ]
    ).unsqueeze(1)
    frias = torch.stack(
        [
            torch.as_tensor(coldest_mask(h, fraction=fraction), device=device)
            for h in heats
        ]
    ).unsqueeze(1)

    variantes = torch.cat(
        [
            lote,
            lote.masked_fill(quentes, 0.0),
            lote.masked_fill(frias, 0.0),
            lote.masked_fill(~quentes, 0.0),
        ]
    )
    probabilidades = torch.softmax(model(variantes).float(), dim=1).cpu().numpy()
    n = images.shape[0]
    indices = np.asarray(class_indices, dtype=np.int64)
    completo, sem_quente, sem_frio, so_quente = (
        probabilidades[i * n : (i + 1) * n][np.arange(n), indices] for i in range(4)
    )
    return tuple(
        Faithfulness(
            baseline=float(completo[i]),
            without_hot=float(sem_quente[i]),
            without_cold=float(sem_frio[i]),
            only_hot=float(so_quente[i]),
        )
        for i in range(n)
    )


# ===================== varredura do conjunto inteiro ========================


@dataclass(frozen=True)
class SaliencyScan:
    """Probabilidades, verdade e medidas do mapa para um conjunto inteiro.

    Existe para que a resposta a pergunta inversa do enunciado ("o modelo acertou
    olhando para o animal ou para o fundo?") nao dependa das 5 imagens escolhidas
    para a figura. Com as 811 do ``test_public`` medidas, a figura ilustra e a
    distribuicao sustenta.
    """

    probabilities: NDArray[np.float64]
    y_true: NDArray[np.int64]
    statistics: tuple[MapStatistics, ...]

    def predictions(self) -> NDArray[np.int64]:
        return self.probabilities.argmax(axis=1)

    def confidence(self) -> NDArray[np.float64]:
        return self.probabilities.max(axis=1)

    def correct(self) -> NDArray[np.bool_]:
        return self.predictions() == self.y_true

    def measure(self, name: str) -> NDArray[np.float64]:
        """Uma medida do mapa como vetor alinhado com ``y_true``."""
        return np.array([float(getattr(s, name)) for s in self.statistics])


@torch.no_grad()
def scan(
    cam: CamModel,
    loader: DataLoader,
    *,
    device: str,
    size: int,
    top_fraction: float = TOP_MASS_FRACTION,
) -> SaliencyScan:
    """Um forward pelo conjunto: probabilidades, verdade e medidas do mapa previsto.

    O mapa medido e sempre o da **classe prevista** — a pergunta e para onde o modelo
    olhou ao decidir o que decidiu, nao onde estaria a resposta certa.

    Os logits saem de ``logits_from_features``, e nao de um segundo forward pelo
    modelo inteiro: dois forwards custariam o dobro e abririam espaco para medir o
    mapa de uma predicao e reportar outra.
    """
    cam.features.eval()
    probabilidades: list[NDArray[np.float64]] = []
    verdades: list[NDArray[np.int64]] = []
    medidas: list[MapStatistics] = []
    for images, labels in loader:
        features = cam.forward_features(images.to(device))
        logits = cam.logits_from_features(features).float()
        probs = torch.softmax(logits, dim=1)
        previstas = probs.argmax(dim=1)
        calor = heatmaps(
            class_activation_maps(features, cam.weight, previstas), size=size
        )
        medidas.extend(
            map_statistics(calor[i], top_fraction=top_fraction)
            for i in range(calor.shape[0])
        )
        probabilidades.append(probs.cpu().numpy().astype(np.float64))
        verdades.append(np.asarray(labels, dtype=np.int64))
    return SaliencyScan(
        probabilities=np.concatenate(probabilidades),
        y_true=np.concatenate(verdades),
        statistics=tuple(medidas),
    )


@torch.no_grad()
def check_decomposition(
    model: nn.Module, cam: CamModel, example: Tensor, *, tolerance: float = 1e-4
) -> float:
    """Confere que ``features -> GAP -> linear`` reproduz ``model(x)``, e devolve o erro.

    Vale rodar em producao e nao so no teste. O mapa CAM e uma afirmacao sobre **este**
    modelo; se a decomposicao divergir do forward original — por uma camada no meio que
    ``children()`` nao capturou na ordem esperada, por exemplo — o mapa explicaria um
    modelo que nao e o avaliado, e a figura sairia plausivel e errada. Custa um forward.
    """
    model.eval()
    cam.features.eval()
    esperado = model(example).float()
    obtido = cam.logits_from_features(cam.forward_features(example)).float()
    diferenca = float((esperado - obtido).abs().max())
    if diferenca > tolerance:
        raise SaliencyError(
            f"a decomposicao para CAM nao reproduz o modelo: diferenca maxima de "
            f"{diferenca:.2e} nos logits (tolerancia {tolerance:.0e}). O mapa "
            f"explicaria um modelo diferente do avaliado."
        )
    return diferenca
