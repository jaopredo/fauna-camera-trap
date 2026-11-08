"""Teto de parametros do modelo entregue.

O enunciado fixa a contagem exata (secao 3): ``sum(p.numel() for p in
model.parameters())`` <= 12 milhoes. Fica num modulo proprio porque e verificado em
tres lugares diferentes — na construcao do modelo, no teste automatizado e no
``predict.py``, que precisa imprimir o numero.
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

from src.errors import AppError

#: Teto do enunciado para o modelo entregue.
MAX_PARAMETERS: Final = 12_000_000


@runtime_checkable
class HasParameters(Protocol):
    """Tudo que exponha ``parameters()`` — qualquer ``nn.Module``.

    Protocolo em vez de ``nn.Module`` para que a contagem seja testavel sem
    construir um modelo de verdade.
    """

    def parameters(self): ...


class ParameterBudgetError(AppError, RuntimeError):
    """Modelo acima do teto de parametros do enunciado."""


def count_parameters(model: HasParameters) -> int:
    """Conta os parametros exatamente como o enunciado manda."""
    return sum(p.numel() for p in model.parameters())


def assert_within_budget(model: HasParameters, *, limit: int = MAX_PARAMETERS) -> int:
    """Confere o teto e devolve a contagem.

    Falhar na construcao e o ponto: um modelo acima do teto treinado por horas
    nao pode ser entregue, e o erro so apareceria na conferencia final.
    """
    total = count_parameters(model)
    if total > limit:
        raise ParameterBudgetError(
            f"modelo com {total:,} parametros excede o teto de {limit:,} "
            f"do enunciado (secao 3)"
        )
    return total
