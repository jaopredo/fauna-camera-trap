"""Comparar uma medida entre dois grupos de imagens, controlando a classe.

Existe por causa de um erro que este projeto cometeu e mediu. Ao testar a hipotese
"o professor erra mais a noite" (Parte 1, trilha C), a comparacao bruta dizia que ele
acertava **12,8% mais** no escuro. Controlando a classe, acertava **6,5% menos**. O
motivo: as imagens noturnas sao dominadas por ``american_black_bear``, que e a melhor
classe do professor, entao a media noturna media uma populacao diferente da diurna, e
nao o efeito da noite. Ver ``docs/decisions.md``, D19.

A licao e geral e nao tem nada de trilha: **sempre que dois grupos de imagens tem
composicao de classes diferente, a media bruta de qualquer medida sobre eles compara
populacoes, nao o efeito do grupo.** A Parte 3 cai no mesmo buraco de forma ainda mais
direta — os erros do modelo se concentram nas classes dificeis, entao dizer "nos erros
o mapa fica mais na borda" sem controlar a classe pode estar so dizendo "as classes
dificeis sao diferentes".

Por isso este modulo e **neutro** e nao vive em nenhuma trilha: a trilha C o usa para
acuracia (que e a medida 0/1 "acertou"), e a Parte 3 o usa para as medidas do mapa de
saliencia. Uma implementacao, dois usos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from src.errors import AppError

#: Minimo de imagens que uma classe precisa ter **em cada grupo** para entrar na
#: comparacao controlada. Uma classe com 2 imagens de um lado produz uma media de 0%
#: ou 100% que domina o resultado sem significar nada.
MIN_SUPPORT = 5


class GroupError(AppError, ValueError):
    """Tamanhos incompativeis entre medida, rotulos e mascara de grupo."""


@dataclass(frozen=True)
class GroupComparison:
    """A mesma medida vista de dois modos, porque **eles discordam**.

    ``raw_delta()`` e a leitura ingenua: media no grupo A menos media no grupo B.
    ``within_class_delta`` compara so dentro de cada classe com suporte nos dois
    lados, e depois tira a media ponderada pelas imagens — e e esse que responde a
    pergunta, porque e o unico em que os dois lados tem a mesma composicao.
    """

    #: nome da medida comparada, para a tabela ("acuracia", "massa_na_borda", ...)
    measure: str
    group_names: tuple[str, str]
    raw_means: tuple[float, float]
    raw_counts: tuple[int, int]
    #: (classe, n_no_grupo_a, media_a, n_no_grupo_b, media_b)
    per_class: tuple[tuple[str, int, float, int, float], ...]
    within_class_delta: float
    comparable_classes: int
    total_classes: int

    def raw_delta(self) -> float:
        return self.raw_means[0] - self.raw_means[1]

    def has_control(self) -> bool:
        """Houve ao menos uma classe com suporte nos dois grupos."""
        return self.comparable_classes > 0

    def controlled_text(self, fmt: str = "{:+.4f}") -> str:
        """O delta controlado, ou ``—`` quando nenhuma classe pode ser comparada.

        Sem classe comparavel, ``within_class_delta`` vale 0,0 — que e a ausencia de
        medida, e nao a medida de uma ausencia. Imprimir "+0,0000" ali faria o leitor
        concluir "nao ha diferenca" onde o certo e "nao da para dizer". A mesma
        distincao que a curva da Parte 2 faz entre desvio zero e desvio nao medido.
        """
        if not self.has_control():
            return "— (nenhuma classe com suporte nos dois grupos)"
        return fmt.format(self.within_class_delta)

    def confounded(self) -> bool:
        """Verdadeiro quando o bruto e o controlado apontam para lados opostos.

        Quando isto da verdadeiro, reportar so o numero bruto seria afirmar o
        contrario do que os dados dizem.
        """
        return self.has_control() and self.raw_delta() * self.within_class_delta < 0


def compare_measure(
    values: Sequence[float] | NDArray[np.float64],
    y_true: Sequence[int] | NDArray[np.int64],
    group_mask: Sequence[bool] | NDArray[np.bool_],
    *,
    class_names: Sequence[str],
    group_names: tuple[str, str] = ("grupo", "resto"),
    min_support: int = MIN_SUPPORT,
    measure: str = "media",
) -> GroupComparison:
    """Media de ``values`` dentro e fora de ``group_mask``, crua e por classe.

    ``values`` e uma medida por imagem — qualquer uma: acerto (0 ou 1), concentracao
    do mapa de saliencia, saturacao. ``y_true`` entra apenas para controlar a classe;
    esta funcao nao avalia modelo nenhum.
    """
    medida = np.asarray(values, dtype=np.float64).reshape(-1)
    verdade = np.asarray(y_true, dtype=np.int64).reshape(-1)
    mascara = np.asarray(group_mask, dtype=bool).reshape(-1)
    if not (medida.size == verdade.size == mascara.size):
        raise GroupError(
            f"tamanhos incompativeis: {medida.size} medidas, {verdade.size} rotulos, "
            f"{mascara.size} no grupo"
        )

    bruto = tuple(
        float(medida[m].mean()) if m.any() else 0.0 for m in (mascara, ~mascara)
    )
    contagens = (int(mascara.sum()), int((~mascara).sum()))

    linhas: list[tuple[str, int, float, int, float]] = []
    deltas: list[float] = []
    pesos: list[int] = []
    for c, nome in enumerate(class_names):
        dentro, fora = (verdade == c) & mascara, (verdade == c) & ~mascara
        if dentro.sum() < min_support or fora.sum() < min_support:
            continue
        a, b = float(medida[dentro].mean()), float(medida[fora].mean())
        linhas.append((nome, int(dentro.sum()), a, int(fora.sum()), b))
        deltas.append(a - b)
        pesos.append(int(dentro.sum() + fora.sum()))

    return GroupComparison(
        measure=measure,
        group_names=group_names,
        raw_means=(bruto[0], bruto[1]),
        raw_counts=contagens,
        per_class=tuple(linhas),
        within_class_delta=float(np.average(deltas, weights=pesos)) if deltas else 0.0,
        comparable_classes=len(linhas),
        total_classes=len(class_names),
    )
