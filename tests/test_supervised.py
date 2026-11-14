"""A estrategia supervisionada — o piso da Parte 0."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.config.schema import StrategyConfig
from src.data.loaders import LABELED
from src.methods import StepOutput, TrainingStrategy, build_strategy

NUM_CLASSES = 21


class ModeloLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, NUM_CLASSES)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return {LABELED: (torch.randn(8, 4), torch.randint(0, NUM_CLASSES, (8,)))}


def test_esta_registrada():
    estrategia = build_strategy(StrategyConfig(name="supervised"))
    assert isinstance(estrategia, TrainingStrategy)


def test_consome_apenas_a_fonte_rotulada():
    """A Parte 0 nao olha o pool sem rotulo — e isso que a torna o piso."""
    estrategia = build_strategy(StrategyConfig(name="supervised"))
    assert estrategia.required_sources == (LABELED,)
    assert estrategia.primary_source == LABELED


def test_passo_devolve_perda_derivavel_e_componente(batch):
    estrategia = build_strategy(StrategyConfig(name="supervised"))
    saida = estrategia.training_step(ModeloLinear(), batch, device="cpu")

    assert isinstance(saida, StepOutput)
    assert saida.loss.ndim == 0
    assert saida.loss.requires_grad, "a perda precisa manter o grafo para o backward"
    assert saida.components["ce"] == pytest.approx(float(saida.loss.detach()))


def test_perda_inicial_proxima_de_log_21(batch):
    """Com logits quase uniformes, a entropia cruzada parte de ln(21) ~ 3,04.

    Serve de sanidade: um valor muito diferente indicaria rotulo fora de faixa ou
    numero de classes errado.
    """
    modelo = ModeloLinear()
    nn.init.zeros_(modelo.fc.weight)
    nn.init.zeros_(modelo.fc.bias)
    estrategia = build_strategy(StrategyConfig(name="supervised"))
    perda = estrategia.training_step(modelo, batch, device="cpu").loss.detach()
    assert float(perda) == pytest.approx(torch.log(torch.tensor(21.0)).item(), abs=1e-5)


def test_label_smoothing_vem_do_config(batch):
    com = build_strategy(
        StrategyConfig(name="supervised", params={"label_smoothing": 0.1})
    )
    sem = build_strategy(StrategyConfig(name="supervised"))
    modelo = ModeloLinear()
    perda_com = float(com.training_step(modelo, batch, device="cpu").loss.detach())
    perda_sem = float(sem.training_step(modelo, batch, device="cpu").loss.detach())
    assert perda_com != pytest.approx(perda_sem)
