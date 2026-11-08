"""Metricas: o F1 macro precisa ser sempre sobre as 21 classes."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    MetricsError,
    accuracy,
    classification_report,
    confusion_matrix,
    macro_f1,
    per_class_f1,
)

NUM_CLASSES = 21


def test_classe_ausente_entra_na_media_como_zero():
    """O caso que motiva o modulo: cavalo tem 2 imagens no val e 4 no test_public.

    Se o modelo nunca preve cavalo e cavalo nao esta no gabarito deste recorte, a
    media precisa continuar dividindo por 21 — nao por 20.
    """
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    confusion = confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES)

    f1_por_classe = per_class_f1(confusion)
    assert f1_por_classe.shape == (NUM_CLASSES,)
    assert f1_por_classe[0] == pytest.approx(1.0)
    assert f1_por_classe[1] == pytest.approx(1.0)
    assert f1_por_classe[2:].sum() == 0.0

    # 2 classes perfeitas em 21: a media e 2/21, nao 1.0
    assert macro_f1(confusion) == pytest.approx(2 / NUM_CLASSES)
    assert accuracy(confusion) == pytest.approx(1.0)


def test_f1_por_classe_com_erro_parcial():
    # classe 0: 2 verdadeiros, 1 acerto, 1 falso positivo da classe 1
    #   precisao 1/2, cobertura 1/2 -> F1 = 0.5
    # classe 1: 1 verdadeiro, 0 acerto, 1 predicao errada -> F1 = 0
    y_true = [0, 0, 1]
    y_pred = [0, 1, 0]
    confusion = confusion_matrix(y_true, y_pred, num_classes=3)
    f1 = per_class_f1(confusion)
    assert f1[0] == pytest.approx(0.5)
    assert f1[1] == pytest.approx(0.0)
    assert macro_f1(confusion) == pytest.approx(0.5 / 3)


def test_matriz_de_confusao_e_linha_verdadeiro_coluna_previsto():
    confusion = confusion_matrix([0, 0, 1], [0, 1, 1], num_classes=2)
    assert confusion.tolist() == [[1, 1], [0, 1]]


def test_matriz_vazia_nao_divide_por_zero():
    confusion = confusion_matrix([], [], num_classes=NUM_CLASSES)
    assert confusion.sum() == 0
    assert macro_f1(confusion) == 0.0
    assert accuracy(confusion) == 0.0


def test_indice_fora_da_faixa_e_recusado():
    with pytest.raises(MetricsError, match="fora da faixa"):
        confusion_matrix([0, 21], [0, 0], num_classes=NUM_CLASSES)


def test_tamanhos_diferentes_sao_recusados():
    with pytest.raises(MetricsError, match="tamanhos diferentes"):
        confusion_matrix([0, 1], [0], num_classes=NUM_CLASSES)


def test_relatorio_traz_suporte_por_classe():
    report = classification_report([0, 0, 1], [0, 0, 1], num_classes=NUM_CLASSES)
    assert report.num_classes == NUM_CLASSES
    assert report.support[0] == 2
    assert report.support[1] == 1
    assert sum(report.support) == 3
    assert report.as_dict()["macro_f1"] == pytest.approx(2 / NUM_CLASSES)


def test_desbalanceamento_puxa_f1_macro_para_baixo_mas_nao_a_acuracia():
    """O argumento do enunciado para usar F1 macro: acuracia esconde classe rara."""
    # 100 imagens da classe 0 (todas certas) e 4 da classe 13 (todas erradas)
    y_true = np.array([0] * 100 + [13] * 4)
    y_pred = np.array([0] * 104)
    report = classification_report(y_true, y_pred, num_classes=NUM_CLASSES)
    assert report.accuracy == pytest.approx(100 / 104)
    assert report.per_class_f1[13] == 0.0
    assert report.macro_f1 < 0.05
