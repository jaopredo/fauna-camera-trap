"""Comparar uma medida entre dois grupos controlando a classe.

O caso que justifica o modulo e o confundidor: dois grupos com composicao de classes
diferente produzem uma diferenca bruta que nao vem do grupo. Estes testes montam
exatamente essa situacao e exigem que as duas leituras apontem para lados opostos.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.groups import GroupError, compare_measure

NOMES = ("urso", "corvo", "passaro")


def montar_confundido():
    """Grupo A parece melhor no bruto e e pior dentro de cada classe.

    O grupo A e dominado por urso, que tem medida alta em qualquer grupo; o B, por
    corvo, que tem medida baixa. Dentro de cada classe, porem, A e pior que B.
    """
    valores: list[float] = []
    verdade: list[int] = []
    grupo: list[bool] = []

    def acrescentar(classe, no_grupo_a, medida, quantas):
        valores.extend([medida] * quantas)
        verdade.extend([classe] * quantas)
        grupo.extend([no_grupo_a] * quantas)

    acrescentar(0, True, 0.80, 20)  # urso no grupo A: alto, mas pior que no B
    acrescentar(0, False, 0.90, 6)
    acrescentar(1, True, 0.10, 6)  # corvo no grupo A: pior que no B
    acrescentar(1, False, 0.20, 20)
    acrescentar(2, True, 0.50, 2)  # suporte pequeno dos dois lados: descartado
    acrescentar(2, False, 0.50, 2)
    return np.array(valores), np.array(verdade), np.array(grupo)


def test_o_bruto_e_o_controlado_podem_apontar_para_lados_opostos():
    valores, verdade, grupo = montar_confundido()
    c = compare_measure(
        valores, verdade, grupo, class_names=NOMES, group_names=("A", "B")
    )
    assert c.raw_delta() > 0, "no bruto o grupo A parece melhor"
    assert c.within_class_delta < 0, "dentro de cada classe o grupo A e pior"
    assert c.confounded(), "o metodo tem de sinalizar a inversao"
    assert c.raw_counts == (28, 28)


def test_classes_sem_suporte_nos_dois_lados_saem_da_conta_controlada():
    valores, verdade, grupo = montar_confundido()
    c = compare_measure(valores, verdade, grupo, class_names=NOMES, min_support=5)
    assert (c.comparable_classes, c.total_classes) == (2, 3)
    assert all(linha[1] >= 5 and linha[3] >= 5 for linha in c.per_class)


def test_min_support_alto_zera_a_comparacao_sem_quebrar():
    valores, verdade, grupo = montar_confundido()
    c = compare_measure(valores, verdade, grupo, class_names=NOMES, min_support=1000)
    assert c.comparable_classes == 0
    assert c.within_class_delta == 0.0
    assert not c.confounded(), "sem classe comparavel nao ha inversao a declarar"


def test_grupo_vazio_nao_divide_por_zero():
    c = compare_measure(
        [0.5, 0.5], [0, 0], [False, False], class_names=NOMES, min_support=1
    )
    assert c.raw_means == (0.0, 0.5)
    assert c.raw_counts == (0, 2)


def test_medida_0_ou_1_e_a_media_da_acuracia():
    """A acuracia e um caso particular: a medida vale 1 quando acertou."""
    c = compare_measure(
        [1.0, 0.0, 1.0, 1.0],
        [0, 0, 1, 1],
        [True, True, False, False],
        class_names=NOMES,
        min_support=1,
        measure="acuracia",
    )
    assert c.raw_means == pytest.approx((0.5, 1.0))
    assert c.measure == "acuracia"


def test_tamanhos_incompativeis_sao_recusados():
    with pytest.raises(GroupError):
        compare_measure([0.1, 0.2], [0], [True, False], class_names=NOMES)


def test_sem_classe_comparavel_o_texto_diz_que_nao_da_para_medir():
    """Zero sem controle e ausencia de medida, nao medida de ausencia."""
    valores, verdade, grupo = montar_confundido()
    c = compare_measure(valores, verdade, grupo, class_names=NOMES, min_support=1000)
    assert not c.has_control()
    assert "—" in c.controlled_text()
    assert "+0" not in c.controlled_text()


def test_com_controle_o_texto_respeita_o_formato_pedido():
    valores, verdade, grupo = montar_confundido()
    c = compare_measure(valores, verdade, grupo, class_names=NOMES)
    assert c.controlled_text() == f"{c.within_class_delta:+.4f}"
    assert c.controlled_text("{:+.1%}") == f"{c.within_class_delta:+.1%}"
