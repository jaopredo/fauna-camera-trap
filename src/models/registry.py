"""Factory com Registry de backbones, selecionados por ``model.name`` na configuracao.

A variacao e real e vem do enunciado: a Parte 0 exige duas arquiteturas (uma CNN
pequena do zero e o backbone com pesos de ImageNet) e a Trilha A exige tres
backbones avaliados lado a lado (ImageNet puro, ImageNet + auto-supervisionado,
aleatorio + auto-supervisionado). Cada um entra como um arquivo novo que se
registra, sem editar o laco de treino nem a avaliacao.

Nenhum backbone concreto esta registrado ainda: a escolha e decisao de modelagem.
Quando houver, o modulo correspondente e importado em ``src/models/__init__.py``
para que o decorador rode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping

from src.config.schema import ModelConfig
from src.errors import AppError
from src.models.budget import assert_within_budget

if TYPE_CHECKING:  # evita importar torch so para anotar tipo
    from torch import nn

#: Um construtor recebe o numero de classes, se usa pesos de ImageNet e os
#: parametros proprios do YAML; devolve o modelo pronto.
ModelFactory = Callable[..., "nn.Module"]

_REGISTRY: dict[str, ModelFactory] = {}


class UnknownModelError(AppError, ValueError):
    """Nome de backbone que ninguem registrou."""


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Decorador que registra um construtor de backbone sob ``name``."""

    def decorator(builder: ModelFactory) -> ModelFactory:
        if name in _REGISTRY:
            raise ValueError(f"backbone ja registrado: {name!r}")
        _REGISTRY[name] = builder
        return builder

    return decorator


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_model(
    cfg: ModelConfig,
    *,
    num_classes: int,
    enforce_budget: bool = True,
    **extra: Any,
) -> "nn.Module":
    """Instancia o backbone declarado em ``cfg.name``.

    ``enforce_budget`` fica ligado por padrao: o teto de 12M e verificado na
    construcao. Desligar so faz sentido para um professor da Trilha C, que nao
    faz parte do modelo entregue.
    """
    try:
        builder = _REGISTRY[cfg.name]
    except KeyError:
        raise UnknownModelError(
            f"model.name desconhecido: {cfg.name!r}. "
            f"Disponiveis: {list(available_models())}"
        ) from None
    model = builder(
        num_classes=num_classes,
        pretrained=cfg.pretrained,
        **{**dict(cfg.params), **extra},
    )
    if enforce_budget:
        assert_within_budget(model)
    return model


def describe(model: "nn.Module", params: Mapping[str, Any] | None = None) -> str:
    """Linha de log com a arquitetura e a contagem de parametros."""
    from src.models.budget import count_parameters

    total = count_parameters(model)
    extra = f" | {dict(params)}" if params else ""
    return f"{type(model).__name__} | {total:,} parametros{extra}"
