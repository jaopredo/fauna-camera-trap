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


# ===================== modelos concretos da Parte 0 =========================
# Importados pelo pacote: src/models/__init__.py registra small_cnn e resnet18.

SMALL_CNN_PARAMETROS = 394_293
RESNET18_PARAMETROS = 11_187_285


def test_os_dois_modelos_do_piso_estao_registrados():
    assert {"small_cnn", "resnet18"} <= set(available_models())


def test_small_cnn_tem_a_contagem_esperada():
    modelo = build_model(ModelConfig(name="small_cnn"), num_classes=21)
    assert count_parameters(modelo) == SMALL_CNN_PARAMETROS


def test_resnet18_cabe_no_teto_do_enunciado():
    """93% do teto: a margem e pequena e qualquer acrescimo precisa ser medido.

    O numero esta travado aqui de proposito. Se uma mudanca futura (outra cabeca,
    uma camada a mais) alterar a contagem, este teste falha antes do treino.
    """
    modelo = build_model(ModelConfig(name="resnet18"), num_classes=21)
    total = count_parameters(modelo)
    assert total == RESNET18_PARAMETROS
    assert total <= MAX_PARAMETERS
    assert total / MAX_PARAMETERS > 0.9, "se a margem aumentou, revise o teste"


@pytest.mark.parametrize("nome", ["small_cnn", "resnet18"])
def test_forward_devolve_logits_das_21_classes(nome):
    modelo = build_model(ModelConfig(name=nome), num_classes=21)
    modelo.eval()
    with torch.no_grad():
        saida = modelo(torch.randn(2, 3, 224, 224))
    assert saida.shape == (2, 21)


def test_small_cnn_recusa_pesos_pre_treinados():
    """A versao (a) do enunciado e "treinada do zero" — nao ha variante silenciosa."""
    with pytest.raises(ValueError, match="treinada do zero"):
        build_model(ModelConfig(name="small_cnn", pretrained=True), num_classes=21)


def test_numero_de_classes_vem_do_catalogo_nao_do_codigo():
    modelo = build_model(ModelConfig(name="resnet18"), num_classes=7)
    assert modelo.fc.out_features == 7
