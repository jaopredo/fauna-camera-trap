"""Diagnosticos da trilha C: matriz de ruido, aluno vs professor, confundidores."""

from __future__ import annotations

import numpy as np
import pytest

from src.tracks.c.analysis import (
    NoiseAnalysisError,
    compare_groups,
    confidence_by_correctness,
    noise_matrix,
    student_vs_teacher,
)

NOMES = ("urso", "corvo", "passaro")


def matriz_exemplo():
    # urso: 4 certos; corvo: 2 certos e 2 ditos passaro; passaro: 3 ditos corvo
    verdade = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    dito = [0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 1]
    return noise_matrix(verdade, dito, class_names=NOMES)


def test_a_matriz_normaliza_por_linha():
    """Linha = classe verdadeira. E assim que ela se le como matriz de ruido."""
    m = matriz_exemplo()
    assert m.rates[0, 0] == pytest.approx(1.0)
    assert m.rates[1, 1] == pytest.approx(0.5)
    assert m.rates[1, 2] == pytest.approx(0.5)
    assert np.allclose(m.rates.sum(axis=1), 1.0)


def test_o_suporte_vem_da_verdade_nao_do_professor():
    assert matriz_exemplo().support.tolist() == [4, 4, 3]


def test_a_diagonal_e_a_cobertura_por_classe():
    assert matriz_exemplo().diagonal().tolist() == pytest.approx([1.0, 0.5, 0.0])


def test_acha_as_confusoes_sistematicas_em_ordem():
    """Taxa alta num par especifico e erro sistematico, nao ruido espalhado."""
    piores = matriz_exemplo().worst_confusions(3)
    assert piores[0][:2] == ("passaro", "corvo")
    assert piores[0][2] == pytest.approx(1.0)
    assert [p[2] for p in piores] == sorted((p[2] for p in piores), reverse=True)


def test_acha_as_classes_que_o_professor_nunca_emite():
    """Nenhum aluno aprende do professor uma classe que ele nunca diz."""
    # no exemplo, "urso" e dito 4 vezes e "passaro" so sai como erro de corvo
    assert matriz_exemplo().collapsed_classes() == ()
    # professor que so diz "urso": as outras duas nunca aparecem no pseudo-rotulo
    assert set(
        noise_matrix([0, 1, 2], [0, 0, 0], class_names=NOMES).collapsed_classes()
    ) == {"corvo", "passaro"}


def test_classe_sem_suporte_nao_divide_por_zero():
    m = noise_matrix([0, 0], [0, 1], class_names=NOMES)
    assert m.rates[2].tolist() == [0.0, 0.0, 0.0]


def test_tamanhos_incompativeis_sao_recusados():
    with pytest.raises(NoiseAnalysisError):
        noise_matrix([0, 1], [0], class_names=NOMES)


def test_rotulo_fora_da_faixa_e_recusado():
    with pytest.raises(NoiseAnalysisError):
        noise_matrix([0, 5], [0, 1], class_names=NOMES)


# ===================== calibracao ===========================================


def test_confianca_separa_acerto_de_erro_quando_calibrado():
    probs = np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25]], dtype=np.float32)
    acerto, erro = confidence_by_correctness(probs, [0, 1])
    assert acerto == pytest.approx(0.9)
    assert erro == pytest.approx(0.4)
    assert acerto > erro


def test_professor_confiantemente_errado_nao_separa():
    """Se as duas medias sao iguais, filtrar por confianca nao filtra erro.

    E a previsao que a trilha B deixou (D17), testavel antes de treinar nada.
    """
    probs = np.array([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]], dtype=np.float32)
    acerto, erro = confidence_by_correctness(probs, [0, 1])
    assert acerto == pytest.approx(erro)


def test_tudo_certo_reporta_zero_no_erro_sem_quebrar():
    probs = np.array([[0.9, 0.05, 0.05]], dtype=np.float32)
    assert confidence_by_correctness(probs, [0]) == (pytest.approx(0.9), 0.0)


# ===================== confundidor ==========================================


def montar_confundido():
    """Grupo A parece melhor no bruto, mas e pior dentro de cada classe.

    Reproduz o que medimos de verdade: a 'noite' tem acuracia bruta maior porque e
    dominada pela classe facil, e acuracia menor quando se compara dentro da classe.
    """
    # (classe, acertos, total, esta_no_grupo_A)
    celulas = [
        # classe 0 (facil) domina o grupo A, e e onde ele vai bem
        (0, 8, 10, True),
        # a mesma classe 0 no grupo B vai MELHOR -> dentro da classe, B ganha
        (0, 10, 10, False),
        # classe 1 (dificil) tem quase todo o seu suporte no grupo B
        (1, 3, 30, False),
        (1, 0, 6, True),
    ]
    verdade, pred, grupo = [], [], []
    for classe, acertos, total, no_grupo in celulas:
        for i in range(total):
            verdade.append(classe)
            pred.append(classe if i < acertos else (classe + 1) % 2)
            grupo.append(no_grupo)
    k = 3
    probs = np.zeros((len(verdade), k), dtype=np.float32)
    probs[np.arange(len(pred)), pred] = 1.0
    return probs, np.array(verdade), np.array(grupo)


def test_detecta_quando_o_bruto_e_o_controlado_discordam():
    probs, verdade, grupo = montar_confundido()
    c = compare_groups(
        probs, verdade, grupo, class_names=NOMES, group_names=("noite", "dia")
    )
    assert c.raw_delta() > 0, "no bruto o grupo A parece melhor"
    assert c.within_class_delta < 0, "dentro da classe o grupo A e pior"
    assert c.confounded(), "o metodo tem de sinalizar a inversao"


def test_descarta_classes_sem_suporte_nos_dois_lados():
    """Uma classe com 2 imagens de um lado produz 0% ou 100% e domina a media."""
    probs, verdade, grupo = montar_confundido()
    c = compare_groups(probs, verdade, grupo, class_names=NOMES, min_support=5)
    assert c.comparable_classes == 2
    assert c.total_classes == 3
    assert all(linha[1] >= 5 and linha[3] >= 5 for linha in c.per_class)


def test_min_support_alto_zera_a_comparacao_sem_quebrar():
    probs, verdade, grupo = montar_confundido()
    c = compare_groups(probs, verdade, grupo, class_names=NOMES, min_support=1000)
    assert c.comparable_classes == 0
    assert c.within_class_delta == 0.0


# ===================== aluno vs professor ===================================


def test_o_ganho_do_aluno_e_por_classe():
    m = matriz_exemplo()
    comparacao = student_vs_teacher(
        class_names=NOMES,
        support=m.support,
        teacher_f1=[0.9, 0.4, 0.0],
        student_f1=[0.8, 0.6, 0.3],
        noise=m,
    )
    assert comparacao.gain().tolist() == pytest.approx([-0.1, 0.2, 0.3])
    assert [n for n, _ in comparacao.student_wins()] == ["passaro", "corvo"]
    assert [n for n, _ in comparacao.teacher_wins()] == ["urso"]


def test_a_pureza_do_rotulo_vem_da_coluna_nao_da_linha():
    """Pureza: das que ele CHAMOU de c, quantas eram c. E o que explica o ganho."""
    m = matriz_exemplo()
    comparacao = student_vs_teacher(
        class_names=NOMES,
        support=m.support,
        teacher_f1=[0.0] * 3,
        student_f1=[0.0] * 3,
        noise=m,
    )
    # corvo foi dito 5 vezes (2 certas), passaro 2 vezes (0 certas)
    assert comparacao.label_purity[1] == pytest.approx(2 / 5)
    assert comparacao.label_purity[2] == pytest.approx(0.0)


def test_as_linhas_da_tabela_tem_as_colunas_do_relatorio():
    m = matriz_exemplo()
    linhas = student_vs_teacher(
        class_names=NOMES,
        support=m.support,
        teacher_f1=[0.9, 0.4, 0.0],
        student_f1=[0.8, 0.6, 0.3],
        noise=m,
    ).as_rows()
    assert set(linhas[0]) == {
        "classe",
        "suporte",
        "f1_professor",
        "f1_aluno",
        "ganho_do_aluno",
        "pureza_do_pseudo_rotulo",
    }
