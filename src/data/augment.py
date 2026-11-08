"""Strategies de aumentacao, selecionadas por nome na configuracao.

Por que registry e nao um ``if`` no dataset: o enunciado pede um experimento de
aumentacao na Trilha A (o random resized crop que corta fora o animal de 2% do
quadro) e o FixMatch precisa de um par fraco/forte da mesma imagem. Sao
estrategias diferentes medidas lado a lado, logo entram como implementacoes
registradas, sem tocar no dataset nem no laco de treino.

Uma estrategia recebe a imagem PIL e devolve o que o modelo consome; para duas
vistas (contrastivo, fraco/forte) devolve uma tupla.

Nenhuma estrategia concreta esta registrada ainda: aumentacao e decisao de
modelagem, e sera definida junto com o backbone.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from src.errors import AppError

#: Assinatura de uma estrategia construida: imagem PIL -> entrada do modelo.
AugmentationStrategy = Callable[[Any], Any]

#: Assinatura de um construtor: parametros -> estrategia de aumentacao.
AugmentationStrategyFactory = Callable[..., AugmentationStrategy]

_REGISTRY: dict[str, AugmentationStrategyFactory] = {}


class UnknownAugmentationError(AppError, ValueError):
    """Nome de estrategia de aumentacao que ninguem registrou."""


def register_augmentation(
    name: str,
) -> Callable[[AugmentationStrategyFactory], AugmentationStrategyFactory]:
    """Decorador que registra um construtor de estrategia sob ``name``."""

    def decorator(builder: AugmentationStrategyFactory) -> AugmentationStrategyFactory:
        if name in _REGISTRY:
            raise ValueError(f"estrategia de aumentacao ja registrada: {name!r}")
        _REGISTRY[name] = builder
        return builder

    return decorator


def available_augmentations() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_augmentation(
    name: str, params: Mapping[str, Any] | None = None, **extra: Any
) -> AugmentationStrategy:
    """Instancia a estrategia de aumentacao registrada sob ``name``.

    ``params`` vem do YAML; ``extra`` carrega o que o chamador sabe e o YAML nao
    (por exemplo ``image_size``, que vem de ``data``).
    """
    try:
        builder = _REGISTRY[name]
    except KeyError:
        raise UnknownAugmentationError(
            f"estrategia de aumentacao desconhecida: {name!r}. "
            f"Disponiveis: {list(available_augmentations())}"
        ) from None
    return builder(**{**(params or {}), **extra})
