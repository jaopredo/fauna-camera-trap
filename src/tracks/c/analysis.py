"""Diagnosticos da trilha C: a matriz de ruido e o aluno contra o professor.

O obrigatorio 2 do enunciado: *"a matriz de confusao do professor e a matriz de
ruido dos pseudo-rotulos. Mostrem-na e digam onde o professor erra de forma
sistematica (noite, animais pequenos, classes parecidas como bird e
american_crow)."*

O obrigatorio 3 termina com uma pergunta: *"em quais classes o aluno supera o
professor, e por que?"*

Funcoes puras sobre matrizes e vetores ja calculados — nao conhecem modelo, loader
nem matplotlib. Nada aqui importa de outra trilha.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from src.analysis.groups import MIN_SUPPORT, GroupComparison, compare_measure
from src.errors import AppError


class NoiseAnalysisError(AppError, ValueError):
    """Entradas incompativeis na analise de ruido."""


@dataclass(frozen=True)
class NoiseMatrix:
    """A matriz de ruido: linha = classe verdadeira, coluna = o que o professor disse.

    Normalizada por linha, e assim que ela se le como ruido: a linha ``c`` diz
    "das imagens que **sao** ``c``, que fracao o professor chamou de cada coisa".
    A diagonal e a cobertura por classe; o resto e para onde o erro vaza.
    """

    #: contagens cruas (num_classes, num_classes)
    counts: NDArray[np.int64]
    #: normalizada por linha; linha sem suporte fica zerada
    rates: NDArray[np.float64]
    support: NDArray[np.int64]
    class_names: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def diagonal(self) -> NDArray[np.float64]:
        """Fracao acertada por classe — a 'pureza' do rotulo daquela classe."""
        return np.diagonal(self.rates).copy()

    def worst_confusions(self, n: int = 10) -> list[tuple[str, str, float, int]]:
        """As ``n`` maiores confusoes fora da diagonal.

        Devolve ``(verdadeira, dita_pelo_professor, taxa, contagem)``. E a lista que
        responde "onde o professor erra de forma sistematica": uma taxa alta num
        par especifico e erro sistematico, nao ruido espalhado.
        """
        fora = self.rates.copy()
        np.fill_diagonal(fora, 0.0)
        ordem = np.argsort(-fora, axis=None)[:n]
        pares = []
        for indice in ordem:
            i, j = np.unravel_index(indice, fora.shape)
            if fora[i, j] <= 0:
                break
            pares.append(
                (
                    self.class_names[i],
                    self.class_names[j],
                    float(fora[i, j]),
                    int(self.counts[i, j]),
                )
            )
        return pares

    def collapsed_classes(self) -> tuple[str, ...]:
        """Classes que o professor **nunca** emite.

        Sao as que nenhum aluno destilado conseguira aprender do professor: o sinal
        simplesmente nao esta nos pseudo-rotulos.
        """
        emitidas = self.counts.sum(axis=0)
        return tuple(nome for nome, n in zip(self.class_names, emitidas) if n == 0)


def noise_matrix(
    y_true: Sequence[int] | NDArray[np.int64],
    y_teacher: Sequence[int] | NDArray[np.int64],
    *,
    class_names: Sequence[str],
) -> NoiseMatrix:
    """Monta a matriz de ruido a partir da verdade e do que o professor disse."""
    verdade = np.asarray(y_true, dtype=np.int64).reshape(-1)
    dito = np.asarray(y_teacher, dtype=np.int64).reshape(-1)
    k = len(class_names)
    if verdade.size != dito.size:
        raise NoiseAnalysisError(
            f"{verdade.size} rotulos verdadeiros para {dito.size} do professor"
        )
    if verdade.size and (
        verdade.max() >= k or dito.max() >= k or verdade.min() < 0 or dito.min() < 0
    ):
        raise NoiseAnalysisError(f"rotulo fora da faixa 0..{k - 1}")

    counts = np.zeros((k, k), dtype=np.int64)
    np.add.at(counts, (verdade, dito), 1)
    support = counts.sum(axis=1)
    rates = np.divide(
        counts,
        support[:, None],
        out=np.zeros((k, k), dtype=np.float64),
        where=support[:, None] > 0,
    )
    return NoiseMatrix(
        counts=counts,
        rates=rates,
        support=support,
        class_names=tuple(class_names),
    )


@dataclass(frozen=True)
class StudentVsTeacher:
    """Por classe: o professor, o aluno, e a diferenca.

    Responde o obrigatorio 3 com numeros em vez de narrativa.
    """

    class_names: tuple[str, ...]
    support: NDArray[np.int64]
    teacher_f1: NDArray[np.float64]
    student_f1: NDArray[np.float64]
    #: fracao dos pseudo-rotulos do professor naquela classe que estao corretos
    label_purity: NDArray[np.float64]

    def gain(self) -> NDArray[np.float64]:
        return self.student_f1 - self.teacher_f1

    def student_wins(self) -> list[tuple[str, float]]:
        """Classes em que o aluno supera o professor, da maior vantagem para baixo."""
        ganho = self.gain()
        ordem = np.argsort(-ganho)
        return [(self.class_names[i], float(ganho[i])) for i in ordem if ganho[i] > 0]

    def teacher_wins(self) -> list[tuple[str, float]]:
        ganho = self.gain()
        ordem = np.argsort(ganho)
        return [(self.class_names[i], float(ganho[i])) for i in ordem if ganho[i] < 0]

    def as_rows(self) -> list[dict[str, object]]:
        return [
            {
                "classe": nome,
                "suporte": int(self.support[i]),
                "f1_professor": float(self.teacher_f1[i]),
                "f1_aluno": float(self.student_f1[i]),
                "ganho_do_aluno": float(self.gain()[i]),
                "pureza_do_pseudo_rotulo": float(self.label_purity[i]),
            }
            for i, nome in enumerate(self.class_names)
        ]


def student_vs_teacher(
    *,
    class_names: Sequence[str],
    support: Sequence[int] | NDArray[np.int64],
    teacher_f1: Sequence[float] | NDArray[np.float64],
    student_f1: Sequence[float] | NDArray[np.float64],
    noise: NoiseMatrix,
) -> StudentVsTeacher:
    """Junta os dois lados e a pureza do rotulo de cada classe.

    ``label_purity[c]`` e a precisao do professor na classe ``c`` **vista pela
    coluna**: das imagens que ele chamou de ``c``, quantas eram ``c``. E a
    quantidade que explica o ganho do aluno: onde o rotulo e puro mas escasso, o
    aluno tem chance de generalizar; onde o rotulo e impuro, ele herda o erro.
    """
    emitidas = noise.counts.sum(axis=0)
    corretas = np.diagonal(noise.counts)
    purity = np.divide(
        corretas,
        emitidas,
        out=np.zeros(len(class_names), dtype=np.float64),
        where=emitidas > 0,
    )
    return StudentVsTeacher(
        class_names=tuple(class_names),
        support=np.asarray(support, dtype=np.int64),
        teacher_f1=np.asarray(teacher_f1, dtype=np.float64),
        student_f1=np.asarray(student_f1, dtype=np.float64),
        label_purity=purity,
    )


def confidence_by_correctness(
    probabilities: NDArray[np.float32],
    y_true: Sequence[int] | NDArray[np.int64],
) -> tuple[float, float]:
    """Confianca media do professor quando acerta e quando erra.

    Se as duas forem parecidas, o professor e **mal calibrado**, e filtrar por
    confianca nao separa acerto de erro — e a previsao que a trilha B nos deixou.
    Este par de numeros a testa diretamente, antes de qualquer treino.
    """
    verdade = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if probabilities.shape[0] != verdade.size:
        raise NoiseAnalysisError(
            f"{probabilities.shape[0]} probabilidades para {verdade.size} rotulos"
        )
    confianca = probabilities.max(axis=1)
    acertou = probabilities.argmax(axis=1) == verdade
    media_acerto = float(confianca[acertou].mean()) if acertou.any() else 0.0
    media_erro = float(confianca[~acertou].mean()) if (~acertou).any() else 0.0
    return media_acerto, media_erro


# ===================== onde o professor erra sistematicamente ===============
# O obrigatorio 2 pede tres hipoteses nomeadas: noite, animais pequenos e classes
# parecidas. Duas delas sao mensuraveis com o que o pacote da:
#
# * **classes parecidas** — sai da matriz de ruido (``worst_confusions``);
# * **noite** — as cameras trap noturnas usam infravermelho, e a imagem sai
#   praticamente sem saturacao. A saturacao media dos pixels e um proxy direto,
#   calculado da propria imagem, sem rotulo novo e sem metadado externo.
#
# **Animais pequenos nao e mensuravel aqui.** Precisaria da caixa do animal, e o
# pacote nao traz anotacao de deteccao. Dizer "o professor erra em animais pequenos"
# sem medir seria inventar; o que da para afirmar e que o pre-processamento do CLIP
# faz CenterCrop, logo animais na borda sao descartados antes do modelo ver.

# O proxy de noite em si (``mean_saturation`` e o limiar) mora em
# ``src.analysis.images``, nao aqui: a Parte 3 usa a mesma medida para diagnosticar
# os erros do modelo entregue, e ela nao pertence a trilha nenhuma. Ver o cabecalho
# daquele modulo.


def compare_groups(
    probabilities: NDArray[np.float32],
    y_true: Sequence[int] | NDArray[np.int64],
    group_mask: NDArray[np.bool_],
    *,
    class_names: Sequence[str],
    group_names: tuple[str, str] = ("grupo", "resto"),
    min_support: int = MIN_SUPPORT,
) -> GroupComparison:
    """Compara a **acuracia** do professor dentro e fora de ``group_mask``.

    Acuracia e a medida 0/1 "acertou esta imagem", entao a comparacao em si —
    bruta, por classe e controlada — e a generica de ``src.analysis.groups``, que a
    Parte 3 tambem usa para as medidas do mapa de saliencia. O que pertence a trilha
    e so a traducao de "probabilidades do professor" para "acertou", que e o que esta
    funcao faz antes de delegar.
    """
    verdade = np.asarray(y_true, dtype=np.int64).reshape(-1)
    mask = np.asarray(group_mask, dtype=bool).reshape(-1)
    if probabilities.shape[0] != verdade.size or mask.size != verdade.size:
        raise NoiseAnalysisError(
            f"tamanhos incompativeis: {probabilities.shape[0]} probabilidades, "
            f"{verdade.size} rotulos, {mask.size} no grupo"
        )
    return compare_measure(
        (probabilities.argmax(axis=1) == verdade).astype(np.float64),
        verdade,
        mask,
        class_names=class_names,
        group_names=group_names,
        min_support=min_support,
        measure="acuracia",
    )
