"""Diagnostico dos pseudo-rotulos — o entregavel obrigatorio 1 da Trilha B."""

from __future__ import annotations

import numpy as np
import pytest

from src.tracks.b.pseudo_labels import (
    DEFAULT_TAUS,
    PseudoLabelError,
    bias_ratio,
    label_distribution,
    pseudo_label_curve,
)

K = 3


def probabilidades(linhas: list[list[float]]) -> np.ndarray:
    return np.asarray(linhas, dtype=float)


def test_precisao_e_cobertura_conferidas_a_mao():
    """Quatro imagens, tres classes, um limiar — conta fechada no papel.

        imagem  verdadeiro  previsto  confianca
          0          0          0        0.9     passa, correto
          1          0          1        0.9     passa, errado
          2          1          1        0.9     passa, correto
          3          2          2        0.4     nao passa

    Em tau=0,5: classe 0 emitiu 1 e acertou 1 -> precisao 1,0; de 2 imagens da
    classe 0, acertou 1 -> cobertura 0,5. Classe 1 emitiu 2 e acertou 1 ->
    precisao 0,5; de 1 imagem da classe 1, acertou 1 -> cobertura 1,0. Classe 2
    nao recebeu nada -> zero nas duas.
    """
    probs = probabilidades(
        [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.9, 0.05], [0.3, 0.3, 0.4]]
    )
    curva = pseudo_label_curve(probs, [0, 0, 1, 2], num_classes=K, taus=(0.5,))

    assert curva.precision[0].tolist() == pytest.approx([1.0, 0.5, 0.0])
    assert curva.coverage[0].tolist() == pytest.approx([0.5, 1.0, 0.0])
    assert curva.mask_rate[0] == pytest.approx(3 / 4)
    assert curva.alive_classes[0] == 2
    assert curva.overall_precision[0] == pytest.approx(2 / 3)
    assert curva.support.tolist() == [2, 1, 1]


def test_classe_sem_pseudo_rotulo_entra_como_zero():
    """Mesma convencao do F1 macro: a classe pesa na media, nao some dela."""
    probs = probabilidades([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]])
    curva = pseudo_label_curve(probs, [0, 0], num_classes=K, taus=(0.5,))
    assert curva.precision[0, 2] == 0.0
    assert curva.macro_precision()[0] == pytest.approx(1 / 3)


def test_cobertura_cai_quando_tau_sobe():
    """Subir o limiar so pode tirar imagens, nunca acrescentar."""
    gerador = np.random.default_rng(0)
    bruto = gerador.random((200, K))
    probs = bruto / bruto.sum(axis=1, keepdims=True)
    verdadeiros = gerador.integers(0, K, 200)

    curva = pseudo_label_curve(probs, verdadeiros, num_classes=K, taus=DEFAULT_TAUS)
    assert np.all(np.diff(curva.mask_rate) <= 1e-12)
    assert np.all(np.diff(curva.macro_coverage()) <= 1e-12)


def test_tau_alto_demais_zera_sem_dividir_por_zero():
    """Com 21 classes, tau=0,99 pode nao deixar nenhuma imagem passar."""
    probs = probabilidades([[0.4, 0.3, 0.3], [0.35, 0.35, 0.3]])
    curva = pseudo_label_curve(probs, [0, 1], num_classes=K, taus=(0.99,))
    assert curva.mask_rate[0] == 0.0
    assert curva.alive_classes[0] == 0
    assert curva.overall_precision[0] == 0.0
    assert not np.isnan(curva.precision).any()


def test_classes_vivas_encolhem_com_tau():
    """O colapso de classe que o enunciado manda investigar, em miniatura."""
    probs = probabilidades(
        [[0.99, 0.005, 0.005], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]
    )
    curva = pseudo_label_curve(probs, [0, 0, 1, 2], num_classes=K, taus=(0.5, 0.95))
    assert curva.alive_classes[0] == 3
    assert curva.alive_classes[1] == 1


def test_distribuicao_conta_so_quem_passou():
    probs = probabilidades([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05], [0.4, 0.3, 0.3]])
    assert label_distribution(probs, tau=0.5, num_classes=K).tolist() == [2, 0, 0]
    assert label_distribution(probs, tau=0.99, num_classes=K).tolist() == [0, 0, 0]


def test_razao_de_vies_marca_super_e_sub_representacao():
    """1 e representacao fiel; acima, a classe recebe mais do que merece."""
    razao = bias_ratio([50, 50, 0], [25, 25, 50])
    assert razao[0] == pytest.approx(2.0)
    assert razao[1] == pytest.approx(2.0)
    assert razao[2] == pytest.approx(0.0)


def test_razao_de_vies_ignora_classe_sem_imagem_real():
    assert bias_ratio([10, 0], [10, 0])[1] == 0.0


def test_linhas_tabulares_cobrem_a_grade_inteira():
    probs = probabilidades([[0.9, 0.05, 0.05]])
    curva = pseudo_label_curve(probs, [0], num_classes=K, taus=(0.5, 0.9))
    linhas = curva.as_rows(("a", "b", "c"))
    assert len(linhas) == 2 * K
    assert {linha["tau"] for linha in linhas} == {0.5, 0.9}


def test_forma_incompativel_e_recusada():
    with pytest.raises(PseudoLabelError, match="esperado"):
        pseudo_label_curve(np.zeros((2, 5)), [0, 1], num_classes=K)


def test_quantidade_incompativel_e_recusada():
    with pytest.raises(PseudoLabelError, match="rotulos"):
        pseudo_label_curve(np.zeros((3, K)), [0, 1], num_classes=K)


def test_tau_fora_da_grade_tem_erro_claro():
    curva = pseudo_label_curve(np.ones((1, K)) / K, [0], num_classes=K, taus=(0.5,))
    with pytest.raises(PseudoLabelError, match="nao esta na grade"):
        curva.at(0.95)
