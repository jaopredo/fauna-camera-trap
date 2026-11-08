"""Selecao de modelo — epoca, limiar, hiperparametro, arquitetura.

Todo caminho de selecao do projeto passa por aqui, e toda funcao daqui chama
``ensure_selectable``. E o que concretiza a regra do enunciado (secao 3: nao treinar
nem fazer selecao de modelo no ``test_public``): como o ``EvaluationResult`` carrega
o proprio ``Split``, nao existe forma de escolher algo com um numero de teste sem
que a chamada levante ``SplitPermissionError``.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence, TypeVar

from src.data.splits import ensure_selectable
from src.evaluation.evaluator import EvaluationResult

T = TypeVar("T")


def best_epoch(
    history: Sequence[EvaluationResult], *, monitor: str = "macro_f1"
) -> int:
    """Indice (base 1) da epoca com melhor ``monitor`` na validacao.

    Empate fica com a epoca mais antiga: com ``val`` de 202 imagens, diferenca
    pequena e provavelmente ruido, e o modelo mais cedo decorou menos.
    """
    if not history:
        raise ValueError("historico vazio: nao ha epoca para escolher")
    for result in history:
        ensure_selectable(result.split)
    values = [result.metric(monitor) for result in history]
    return values.index(max(values)) + 1


def best_candidate(
    candidates: Mapping[T, EvaluationResult], *, monitor: str = "macro_f1"
) -> T:
    """Chave do melhor candidato — limiar tau, backbone, hiperparametro.

    Serve a escolha de tau da Trilha B e a comparacao de backbones da Trilha A,
    sempre com numeros de validacao.
    """
    if not candidates:
        raise ValueError("nenhum candidato para escolher")
    for result in candidates.values():
        ensure_selectable(result.split)
    return max(candidates, key=lambda key: candidates[key].metric(monitor))


def is_improvement(
    current: EvaluationResult, best_so_far: float, *, monitor: str = "macro_f1"
) -> bool:
    """Se ``current`` supera o melhor valor visto — usado pelo checkpoint ``best.pt``."""
    ensure_selectable(current.split)
    return current.metric(monitor) > best_so_far


def select_by(
    candidates: Mapping[T, EvaluationResult], key: Callable[[EvaluationResult], float]
) -> T:
    """Variante com criterio proprio (ex.: F1 macro menos desvio entre classes)."""
    if not candidates:
        raise ValueError("nenhum candidato para escolher")
    for result in candidates.values():
        ensure_selectable(result.split)
    return max(candidates, key=lambda name: key(candidates[name]))
