"""Contrato comum das Strategies de treino (L do SOLID).

Roda sobre todas as estrategias registradas: quando a trilha for implementada,
ela entra aqui automaticamente, sem editar o teste.
"""

from __future__ import annotations

import pytest

from src.config.schema import StrategyConfig
from src.data.loaders import LABELED, UNLABELED
from src.methods import (
    StepOutput,
    TrainingStrategy,
    UnknownStrategyError,
    available_strategies,
    build_strategy,
)

FONTES_CONHECIDAS = {LABELED, UNLABELED}


def test_estrategia_desconhecida_lista_as_opcoes():
    with pytest.raises(UnknownStrategyError, match="Disponiveis"):
        build_strategy(StrategyConfig(name="nao_existe"))


def test_contrato_nao_pode_ser_instanciado_sem_training_step():
    with pytest.raises(TypeError):
        TrainingStrategy()  # type: ignore[abstract]


@pytest.mark.parametrize("nome", available_strategies())
def test_estrategia_registrada_cumpre_o_contrato(nome):
    strategy = build_strategy(StrategyConfig(name=nome), num_classes=21)
    assert isinstance(strategy, TrainingStrategy)
    assert strategy.primary_source in strategy.required_sources, (
        f"{nome}: primary_source precisa estar em required_sources"
    )
    desconhecidas = set(strategy.required_sources) - FONTES_CONHECIDAS
    assert not desconhecidas, f"{nome}: fontes desconhecidas {sorted(desconhecidas)}"
    assert isinstance(strategy.state_dict(), dict)


def test_step_output_guarda_componentes_em_float():
    import torch

    saida = StepOutput(loss=torch.tensor(1.0), components={"ce": 1.0})
    assert saida.components["ce"] == pytest.approx(1.0)
