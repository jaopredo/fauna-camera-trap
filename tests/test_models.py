"""Teto de parametros e registry de backbones."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.config.schema import ModelConfig
from src.models import (
    MAX_PARAMETERS,
    ParameterBudgetError,
    UnknownModelError,
    assert_within_budget,
    available_models,
    build_model,
    count_parameters,
    load_checkpoint,
    register_model,
    save_checkpoint,
)
from src.models.checkpoint import CheckpointError, CheckpointState


class TinyModel(nn.Module):
    def __init__(self, num_classes: int = 21, width: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(width, num_classes)

    def forward(self, x):  # pragma: no cover - nao usado nos testes de contagem
        return self.fc(x)


def test_teto_do_enunciado():
    assert MAX_PARAMETERS == 12_000_000


def test_contagem_bate_com_a_formula_do_enunciado():
    modelo = TinyModel(num_classes=21, width=4)
    esperado = sum(p.numel() for p in modelo.parameters())
    assert count_parameters(modelo) == esperado == 4 * 21 + 21


def test_modelo_dentro_do_teto_passa():
    assert assert_within_budget(TinyModel()) > 0


def test_modelo_acima_do_teto_e_recusado():
    grande = nn.Linear(4000, 4000)  # 16.004.000 parametros
    assert count_parameters(grande) > MAX_PARAMETERS
    with pytest.raises(ParameterBudgetError, match="excede o teto"):
        assert_within_budget(grande)


def test_backbone_desconhecido_lista_as_opcoes():
    with pytest.raises(UnknownModelError, match="Disponiveis"):
        build_model(ModelConfig(name="nao_existe"), num_classes=21)


def test_registry_constroi_e_aplica_o_teto():
    nome = "_teste_pequeno"
    register_model(nome)(lambda num_classes, pretrained, **kw: TinyModel(num_classes))
    try:
        assert nome in available_models()
        modelo = build_model(ModelConfig(name=nome), num_classes=21)
        assert isinstance(modelo, TinyModel)

        register_model("_teste_grande")(
            lambda num_classes, pretrained, **kw: nn.Linear(4000, 4000)
        )
        with pytest.raises(ParameterBudgetError):
            build_model(ModelConfig(name="_teste_grande"), num_classes=21)
        # o professor da Trilha C nao e o modelo entregue: pode passar do teto
        assert build_model(
            ModelConfig(name="_teste_grande"), num_classes=21, enforce_budget=False
        )
    finally:
        from src.models import registry

        registry._REGISTRY.pop(nome, None)
        registry._REGISTRY.pop("_teste_grande", None)


def test_checkpoint_sobrevive_a_ida_e_volta(tmp_path):
    modelo = TinyModel()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    estado = CheckpointState(
        epoch=7, best_metric=0.42, monitor="macro_f1", config={"name": "x"}
    )
    caminho = save_checkpoint(
        tmp_path / "best.pt", model=modelo, state=estado, optimizer=otimizador
    )

    outro = TinyModel()
    lido = load_checkpoint(caminho, model=outro)
    assert lido.epoch == 7
    assert lido.best_metric == pytest.approx(0.42)
    assert lido.monitor == "macro_f1"
    assert torch.allclose(outro.fc.weight, modelo.fc.weight)


def test_checkpoint_ausente_tem_erro_claro(tmp_path):
    with pytest.raises(CheckpointError, match="nao encontrado"):
        load_checkpoint(tmp_path / "nao_existe.pt", model=TinyModel())
