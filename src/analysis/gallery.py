"""Quais imagens entram na galeria da Parte 3, e o diagnostico de cada uma.

O enunciado pede "tres erros do modelo final" e "cinco acertos", cada um com imagem,
classe verdadeira, prevista, confianca, mapa de saliencia e **um diagnostico
concreto**. Este modulo responde as duas perguntas que isso levanta: *quais* imagens,
e o que se pode afirmar sobre elas.

**A escolha das imagens e uma regra, nao um gosto.** Escolher a dedo "os erros mais
interessantes" depois de olhar os mapas seria selecionar a evidencia que confirma a
historia. As regras estao em ``select_errors`` e ``select_hits``, sao deterministicas
e saem das probabilidades — qualquer pessoa que rode o script ve as mesmas imagens.

**O diagnostico e montado de medidas, nao de uma descricao do que aparece na foto.**
Isso nao e preciosismo: o ``CLAUDE.md`` proibe usar modelo fechado para classificar
imagem do conjunto, e uma frase como "o mapa esta no tronco da arvore ao fundo" e
exatamente um ato de classificacao. Entao cada etiqueta do diagnostico vem de uma
comparacao mensuravel (saturacao, massa do mapa, oclusao), e a leitura visual final
fica para quem apresenta — que olha a figura e tem o gabarito do pacote na mao.

**Os limiares saem da distribuicao do proprio conjunto**, nao de uma constante
escolhida a mao: "atencao difusa" quer dizer *abaixo da mediana das 811 imagens do
test_public*, e "massa na borda" quer dizer *acima do percentil 75*. Isso evita numero
magico e torna a etiqueta comparativa, que e a unica leitura honesta de uma medida sem
unidade natural. Duas excecoes declaradas, ``LOW_IOU`` e ``MIN_BORDER_EXCESS``, existem
porque um percentil sozinho nao protege contra diferenca irrelevante; as duas aparecem
na tabela com o numero cru ao lado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from src.analysis.images import NIGHT_SATURATION_THRESHOLD
from src.analysis.saliency import Faithfulness, MapStatistics, SaliencyScan
from src.errors import AppError

#: Abaixo desta sobreposicao, os mapas da classe prevista e da verdadeira apontam
#: para regioes diferentes. Um quarto de intersecao sobre uniao e uma convencao
#: declarada, nao uma medida — por isso o numero cru aparece sempre ao lado da
#: etiqueta na tabela, e a etiqueta nunca e o unico registro.
LOW_IOU = 0.25

#: Excesso minimo de massa na borda, em pontos da massa total, para a etiqueta
#: valer. O percentil do conjunto sozinho nao basta: a faixa de borda ocupa ~64% da
#: area, entao um mapa difuso deixa ~64% da massa nela por acidente geometrico, e num
#: conjunto de mapas difusos o percentil 75 cai rente a esse valor. Sem este piso, a
#: etiqueta dispararia com 0,03 ponto de excesso — ruido apresentado como achado.
MIN_BORDER_EXCESS = 0.05

#: Percentis que definem "difuso" e "na borda" em relacao ao conjunto.
DIFFUSE_PERCENTILE = 50
BORDER_PERCENTILE = 75
SMALL_REGION_PERCENTILE = 25


class GalleryError(AppError, ValueError):
    """Entradas inconsistentes na montagem da galeria."""


@dataclass(frozen=True)
class GalleryPick:
    """Uma imagem escolhida, identificada pela **posicao** na ordem do conjunto.

    Posicao e nao ``image_id`` porque a escolha e feita sobre a matriz de
    probabilidades, que nao conhece nome de arquivo; quem traduz posicao em imagem e
    o script, que tem o catalogo. Assim este modulo continua testavel sem dados.
    """

    position: int
    true_index: int
    predicted_index: int
    confidence: float
    #: por que **esta** imagem entrou — vai para a tabela, para a regra ficar visivel
    reason: str

    @property
    def correct(self) -> bool:
        return self.true_index == self.predicted_index


def _check(probabilities: NDArray[np.float64], y_true: NDArray[np.int64]) -> None:
    if probabilities.ndim != 2:
        raise GalleryError(
            f"probabilidades deveriam ser (n,k); veio {probabilities.shape}"
        )
    if probabilities.shape[0] != y_true.size:
        raise GalleryError(
            f"{probabilities.shape[0]} probabilidades para {y_true.size} rotulos"
        )


def select_errors(
    probabilities: NDArray[np.float64],
    y_true: Sequence[int] | NDArray[np.int64],
    *,
    class_names: Sequence[str],
    limit: int = 3,
) -> tuple[GalleryPick, ...]:
    """Os ``limit`` erros da galeria: um por par de confusao mais frequente.

    Duas decisoes, e as duas tem motivo:

    * **um erro por par ``(verdadeira, prevista)``**, e nao os tres erros mais
      confiantes do conjunto. Os tres mais confiantes tenderiam a ser o mesmo par
      repetido (o modelo erra muito nas mesmas duas classes), e a galeria mostraria
      um modo de falha tres vezes em vez de tres modos de falha;
    * **dentro do par, o erro mais confiante.** Um erro em que o modelo hesitou
      (confianca 0,2) nao tem muito a explicar. O erro confiante e o informativo: o
      modelo tinha certeza, e o mapa mostra em que ele se baseou para ter certeza.

    Os pares vem ordenados por frequencia, de forma que o primeiro erro da galeria e a
    confusao mais sistematica do modelo e nao um caso isolado.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    verdade = np.asarray(y_true, dtype=np.int64).reshape(-1)
    _check(probs, verdade)
    previstas = probs.argmax(axis=1)
    confianca = probs.max(axis=1)
    errou = np.flatnonzero(previstas != verdade)

    pares: dict[tuple[int, int], list[int]] = {}
    for i in errou:
        pares.setdefault((int(verdade[i]), int(previstas[i])), []).append(int(i))

    # frequencia primeiro; empate pelo erro mais confiante do par, para a ordem ser
    # total e o resultado nao depender da ordem de insercao do dicionario
    ordenados = sorted(
        pares.items(),
        key=lambda item: (-len(item[1]), -max(confianca[item[1]])),
    )

    escolhas: list[GalleryPick] = []
    for posicao_do_par, ((verdadeira, prevista), indices) in enumerate(
        ordenados[:limit], start=1
    ):
        i = int(max(indices, key=lambda j: confianca[j]))
        escolhas.append(
            GalleryPick(
                position=i,
                true_index=verdadeira,
                predicted_index=prevista,
                confidence=float(confianca[i]),
                reason=(
                    f"{class_names[verdadeira]} -> {class_names[prevista]} e a "
                    f"{posicao_do_par}a confusao mais frequente ({len(indices)} "
                    f"imagens); este e o erro mais confiante do par"
                ),
            )
        )
    return tuple(escolhas)


def select_hits(
    probabilities: NDArray[np.float64],
    y_true: Sequence[int] | NDArray[np.int64],
    *,
    class_names: Sequence[str],
    limit: int = 5,
) -> tuple[GalleryPick, ...]:
    """Os ``limit`` acertos da galeria: os mais confiantes, de classes distintas.

    A pergunta inversa do enunciado e "o modelo acertou olhando para o animal ou para
    o fundo?". O caso que mais importa responder e o **acerto confiante**: se o modelo
    tem 99% de certeza olhando para o fundo, ele acertou pelo motivo errado, e esse
    acerto nao vai se repetir em outra camera. Um acerto hesitante nao sustenta
    afirmacao nenhuma sobre atalho.

    **Classes distintas** porque o ``test_public`` tem 82 imagens de urso e 4 de
    cavalo: os cinco acertos mais confiantes seriam provavelmente cinco ursos, e a
    galeria falaria de uma classe so.

    Isto **nao** e uma amostra representativa, e nao deve ser lida como uma: ela e, de
    proposito, o melhor caso. A representatividade vem de ``compare_measure`` sobre as
    811 imagens, nao destas cinco.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    verdade = np.asarray(y_true, dtype=np.int64).reshape(-1)
    _check(probs, verdade)
    previstas = probs.argmax(axis=1)
    confianca = probs.max(axis=1)
    acertou = np.flatnonzero(previstas == verdade)
    # ordem decrescente de confianca; empate pela posicao, para ser deterministico
    ordem = sorted(acertou, key=lambda i: (-confianca[i], int(i)))

    escolhas: list[GalleryPick] = []
    vistas: set[int] = set()
    for i in ordem:
        classe = int(verdade[i])
        if classe in vistas:
            continue
        vistas.add(classe)
        escolhas.append(
            GalleryPick(
                position=int(i),
                true_index=classe,
                predicted_index=classe,
                confidence=float(confianca[i]),
                reason=(
                    f"acerto mais confiante da classe {class_names[classe]} "
                    f"({len(escolhas) + 1}a classe distinta em ordem de confianca)"
                ),
            )
        )
        if len(escolhas) == limit:
            break
    return tuple(escolhas)


# ===================== o diagnostico ========================================


@dataclass(frozen=True)
class AttentionReference:
    """Os cortes que tornam o diagnostico comparativo em vez de arbitrario.

    Vem da distribuicao das medidas no conjunto inteiro. Dizer "esta atencao e difusa"
    sem referencia nao significa nada — difusa comparada a que? Comparada a mediana
    das 811 imagens significa algo, e o numero que define o corte e reportado junto.
    """

    diffuse_below: float
    border_above: float
    small_region_below: float
    #: quantas imagens formaram a referencia, para o leitor saber o peso dela
    images: int

    @classmethod
    def from_scan(cls, scan: SaliencyScan) -> "AttentionReference":
        if not scan.statistics:
            raise GalleryError("varredura vazia: nao ha de onde tirar a referencia")
        return cls(
            diffuse_below=float(
                np.percentile(scan.measure("concentration"), DIFFUSE_PERCENTILE)
            ),
            border_above=float(
                np.percentile(scan.measure("border_mass"), BORDER_PERCENTILE)
            ),
            small_region_below=float(
                np.percentile(scan.measure("hot_pixels"), SMALL_REGION_PERCENTILE)
            ),
            images=len(scan.statistics),
        )

    def describe(self) -> str:
        return (
            f"{self.images} imagens do conjunto: "
            f"concentracao mediana {self.diffuse_below:.3f}, "
            f"massa na borda no percentil {BORDER_PERCENTILE} "
            f"{self.border_above:.3f}, regiao quente no percentil "
            f"{SMALL_REGION_PERCENTILE} {self.small_region_below:.0f} px"
        )


@dataclass(frozen=True)
class Diagnosis:
    """As etiquetas que as medidas sustentam — e nada alem delas."""

    tags: tuple[str, ...]

    def as_text(self) -> str:
        """Texto para a tabela. Sem etiqueta, diz isso explicitamente.

        "Nada fora do padrao" e um resultado, nao uma falha da analise: significa que
        aquele erro nao se explica por noite, por atencao difusa, por borda nem por
        mapa infiel — e que a explicacao, se houver, esta na aparencia das classes.
        """
        if not self.tags:
            return "nada fora do padrao do conjunto nas medidas disponiveis"
        return "; ".join(self.tags)


def diagnose(
    statistics: MapStatistics,
    reference: AttentionReference,
    *,
    saturation: float | None = None,
    iou_true_vs_predicted: float | None = None,
    probe: Faithfulness | None = None,
) -> Diagnosis:
    """Monta o diagnostico de uma imagem a partir do que foi medido nela.

    Cada etiqueta e uma comparacao, e todas as comparacoes sao reproduziveis. O que
    **nao** esta aqui e igualmente importante: nao ha etiqueta "animal pequeno", porque
    medir o tamanho do animal exigiria a caixa dele e o pacote nao traz anotacao de
    deteccao. O substituto honesto e "regiao usada pequena", que fala do mapa e nao do
    animal.
    """
    etiquetas: list[str] = []
    if saturation is not None and saturation < NIGHT_SATURATION_THRESHOLD:
        etiquetas.append(f"noturna em IR (saturacao media {saturation:.1f})")
    if statistics.concentration < reference.diffuse_below:
        etiquetas.append(
            f"atencao difusa (concentracao {statistics.concentration:.3f} < mediana "
            f"{reference.diffuse_below:.3f} do conjunto)"
        )
    if (
        statistics.border_mass > reference.border_above
        and statistics.border_excess >= MIN_BORDER_EXCESS
    ):
        etiquetas.append(
            f"mapa puxado para a borda ({statistics.border_mass:.1%} da massa numa "
            f"faixa que ocupa {statistics.border_area:.1%} da area, excesso de "
            f"{statistics.border_excess:+.1%})"
        )
    if statistics.hot_pixels < reference.small_region_below:
        etiquetas.append(
            f"regiao usada pequena (~{statistics.hot_side:.0f}x"
            f"{statistics.hot_side:.0f} px de {statistics.side}x{statistics.side})"
        )
    if iou_true_vs_predicted is not None and iou_true_vs_predicted < LOW_IOU:
        etiquetas.append(
            f"olhou para outro lugar: o mapa da classe verdadeira quase nao se "
            f"sobrepoe ao da prevista (IoU {iou_true_vs_predicted:.2f})"
        )
    if probe is not None and not probe.explains():
        etiquetas.append(
            f"o mapa nao explica a decisao: apagar a regiao apontada custa "
            f"{probe.drop_hot():+.3f} de confianca, menos que apagar uma regiao fria "
            f"de area igual ({probe.drop_cold():+.3f})"
        )
    return Diagnosis(tags=tuple(etiquetas))
