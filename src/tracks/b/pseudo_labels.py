"""Qualidade dos pseudo-rotulos em funcao do limiar tau.

E o diagnostico obrigatorio 1 da Trilha B: *"precisao e cobertura dos pseudo-rotulos
por classe, em funcao do limiar tau"*. O enunciado manda medir isso rodando o
pseudo-rotulador sobre o ``test_public`` como se ele nao tivesse rotulo — e deixa
explicito que **e so diagnostico**: a escolha de tau e feita no ``val``.

Funcoes puras sobre probabilidades ja calculadas: nao conhecem modelo, loader nem
matplotlib. Um unico forward guarda a matriz ``(n_imagens, n_classes)`` e a varredura
de tau sai daqui, em memoria, para qualquer grade.

Definicoes, para o pseudo-rotulador que emite rotulo so quando ``max p >= tau``:

``precisao[c]``   dos que receberam pseudo-rotulo ``c``, quantos sao mesmo ``c``
``cobertura[c]``  dos que sao mesmo ``c``, quantos receberam pseudo-rotulo ``c``

Sao precisao e cobertura restritas as imagens que passaram o limiar. Classe sem
nenhum pseudo-rotulo recebe zero nas duas, pela mesma convencao do F1 macro em
``src/metrics.py``: ela precisa pesar na media, nao desaparecer dela.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.errors import AppError

#: Grade padrao de limiares. Comeca em 0,50 porque abaixo disso o argmax de 21
#: classes quase sempre passa, e a curva vira uma reta sem informacao.
DEFAULT_TAUS: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)


class PseudoLabelError(AppError, ValueError):
    """Entrada incompativel com o numero de classes declarado."""


@dataclass(frozen=True)
class PseudoLabelCurve:
    """Precisao e cobertura por classe, para cada limiar da grade."""

    taus: tuple[float, ...]
    #: forma (n_taus, num_classes)
    precision: NDArray[np.float64]
    coverage: NDArray[np.float64]
    #: fracao de TODAS as imagens que receberam algum pseudo-rotulo, por tau
    mask_rate: NDArray[np.float64]
    #: de quantas classes distintas saiu ao menos um pseudo-rotulo, por tau
    alive_classes: NDArray[np.int64]
    #: dos pseudo-rotulos emitidos, fracao correta — a precisao agregada
    overall_precision: NDArray[np.float64]
    support: NDArray[np.int64]

    @property
    def num_classes(self) -> int:
        return self.precision.shape[1]

    def macro_precision(self) -> NDArray[np.float64]:
        """Media sobre **todas** as classes, inclusive as sem pseudo-rotulo."""
        return self.precision.mean(axis=1)

    def macro_coverage(self) -> NDArray[np.float64]:
        return self.coverage.mean(axis=1)

    def at(self, tau: float) -> int:
        """Indice do limiar ``tau`` na grade."""
        try:
            return self.taus.index(tau)
        except ValueError:
            raise PseudoLabelError(
                f"tau {tau} nao esta na grade {list(self.taus)}"
            ) from None

    def as_rows(self, labels: tuple[str, ...]) -> list[dict[str, object]]:
        """Forma tabular, uma linha por (tau, classe) — vira CSV no relatorio."""
        if len(labels) != self.num_classes:
            raise PseudoLabelError(
                f"{len(labels)} nomes para {self.num_classes} classes"
            )
        return [
            {
                "tau": tau,
                "classe": labels[c],
                "suporte": int(self.support[c]),
                "precisao": float(self.precision[t, c]),
                "cobertura": float(self.coverage[t, c]),
            }
            for t, tau in enumerate(self.taus)
            for c in range(self.num_classes)
        ]


def _validate(
    probabilities: ArrayLike, y_true: ArrayLike, num_classes: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    probs = np.asarray(probabilities, dtype=np.float64)
    true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if probs.ndim != 2 or probs.shape[1] != num_classes:
        raise PseudoLabelError(
            f"probabilidades com forma {probs.shape}; esperado (n, {num_classes})"
        )
    if probs.shape[0] != true.size:
        raise PseudoLabelError(
            f"{probs.shape[0]} probabilidades para {true.size} rotulos"
        )
    if true.size and (true.min() < 0 or true.max() >= num_classes):
        raise PseudoLabelError(f"rotulo fora da faixa 0..{num_classes - 1}")
    return probs, true


def pseudo_label_curve(
    probabilities: ArrayLike,
    y_true: ArrayLike,
    *,
    num_classes: int,
    taus: tuple[float, ...] = DEFAULT_TAUS,
) -> PseudoLabelCurve:
    """Varre ``taus`` e devolve precisao e cobertura por classe em cada um."""
    probs, true = _validate(probabilities, y_true, num_classes)
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    support = np.bincount(true, minlength=num_classes).astype(np.int64)

    precision = np.zeros((len(taus), num_classes), dtype=np.float64)
    coverage = np.zeros_like(precision)
    mask_rate = np.zeros(len(taus), dtype=np.float64)
    alive = np.zeros(len(taus), dtype=np.int64)
    overall = np.zeros(len(taus), dtype=np.float64)

    for t, tau in enumerate(taus):
        mask = confidence >= tau
        n_mascarado = int(mask.sum())
        mask_rate[t] = n_mascarado / len(true) if len(true) else 0.0
        if n_mascarado == 0:
            continue

        emitidos = np.bincount(predicted[mask], minlength=num_classes)
        corretos = np.bincount(
            predicted[mask][predicted[mask] == true[mask]], minlength=num_classes
        )
        alive[t] = int((emitidos > 0).sum())
        overall[t] = float(corretos.sum() / emitidos.sum())
        precision[t] = np.divide(
            corretos, emitidos, out=np.zeros(num_classes), where=emitidos > 0
        )
        coverage[t] = np.divide(
            corretos, support, out=np.zeros(num_classes), where=support > 0
        )

    return PseudoLabelCurve(
        taus=tuple(taus),
        precision=precision,
        coverage=coverage,
        mask_rate=mask_rate,
        alive_classes=alive,
        overall_precision=overall,
        support=support,
    )


def label_distribution(
    probabilities: ArrayLike, *, tau: float, num_classes: int
) -> NDArray[np.int64]:
    """Quantos pseudo-rotulos cada classe recebeu no limiar ``tau``.

    E a materia-prima do diagnostico obrigatorio 2: comparada com a distribuicao
    verdadeira, mostra se os pseudo-rotulos estao indo para as classes grandes.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != num_classes:
        raise PseudoLabelError(
            f"probabilidades com forma {probs.shape}; esperado (n, {num_classes})"
        )
    mask = probs.max(axis=1) >= tau
    return np.bincount(probs.argmax(axis=1)[mask], minlength=num_classes).astype(
        np.int64
    )


def bias_ratio(pseudo_counts: ArrayLike, true_counts: ArrayLike) -> NDArray[np.float64]:
    """Razao entre a fatia dos pseudo-rotulos e a fatia real, por classe.

    ``1`` e representacao fiel; acima de 1, a classe recebe mais pseudo-rotulos do
    que merece; abaixo, menos. E um numero so por classe, mais facil de ler na
    apresentacao do que duas distribuicoes lado a lado.
    """
    pseudo = np.asarray(pseudo_counts, dtype=np.float64)
    real = np.asarray(true_counts, dtype=np.float64)
    if pseudo.shape != real.shape:
        raise PseudoLabelError(f"formas diferentes: {pseudo.shape} e {real.shape}")
    fatia_pseudo = pseudo / pseudo.sum() if pseudo.sum() else np.zeros_like(pseudo)
    fatia_real = real / real.sum() if real.sum() else np.zeros_like(real)
    return np.divide(
        fatia_pseudo, fatia_real, out=np.zeros_like(fatia_real), where=fatia_real > 0
    )
