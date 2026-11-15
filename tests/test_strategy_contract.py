"""Contrato comum das Strategies de treino (L do SOLID).

Roda sobre **todas** as estrategias registradas, sem lista de excecoes: uma trilha
nova entra aqui automaticamente, e apagar uma trilha a remove automaticamente.

Algumas estrategias precisam de um artefato externo para nascer — a destilacao da
trilha C precisa do ``.npz`` que o professor gravou. Para essas, o que o teste
compartilhado verifica e que a ausencia produz uma **mensagem util**, nao um
``TypeError`` de Python; o contrato em si e verificado pelo teste da propria trilha,
que sabe montar o artefato. Ver ``tests/contract.py``.
"""

from __future__ import annotations

import pytest

from src.config.schema import StrategyConfig
from src.errors import AppError
from src.methods import (
    StepOutput,
    TrainingStrategy,
    UnknownStrategyError,
    available_strategies,
    build_strategy,
)
from tests.contract import assert_cumpre_o_contrato


def test_estrategia_desconhecida_lista_as_opcoes():
    with pytest.raises(UnknownStrategyError, match="Disponiveis"):
        build_strategy(StrategyConfig(name="nao_existe"))


def test_contrato_nao_pode_ser_instanciado_sem_training_step():
    with pytest.raises(TypeError):
        TrainingStrategy()  # type: ignore[abstract]


def test_ha_estrategias_registradas():
    """Guarda contra suite vazia: sem registro, o teste abaixo nao olharia nada."""
    assert available_strategies()


@pytest.mark.parametrize("nome", available_strategies())
def test_estrategia_registrada_cumpre_o_contrato_ou_explica_o_que_falta(nome):
    try:
        strategy = build_strategy(StrategyConfig(name=nome), num_classes=21)
    except AppError as erro:
        # precisa de artefato externo: a mensagem tem de dizer o que, e como obter
        mensagem = str(erro)
        assert len(mensagem) > 40, (
            f"{nome}: recusou nascer com a mensagem {mensagem!r}, curta demais "
            f"para alguem consertar o YAML"
        )
        return
    assert_cumpre_o_contrato(strategy, nome)


def test_step_output_guarda_componentes_em_float():
    import torch

    saida = StepOutput(loss=torch.tensor(1.0), components={"ce": 1.0})
    assert saida.components["ce"] == pytest.approx(1.0)
