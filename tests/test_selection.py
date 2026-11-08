"""Bloqueio do test_public em todo caminho de selecao de modelo."""

from __future__ import annotations

import pytest

from src.data.splits import Split, SplitPermissionError
from src.evaluation import best_candidate, best_epoch, is_improvement, select_by
from src.evaluation.evaluator import EvaluationResult
from src.metrics import classification_report

NUM_CLASSES = 21


def resultado(split: Split, acertos: int, total: int = 21) -> EvaluationResult:
    """Resultado sintetico com ``acertos`` classes previstas corretamente."""
    y_true = list(range(total))
    y_pred = [i if i < acertos else (i + 1) % total for i in range(total)]
    return EvaluationResult(
        split=split,
        report=classification_report(y_true, y_pred, num_classes=NUM_CLASSES),
    )


def test_melhor_epoca_vem_do_val():
    historico = [resultado(Split.VAL, a) for a in (5, 15, 10)]
    assert best_epoch(historico) == 2


def test_empate_fica_com_a_epoca_mais_antiga():
    """Com val de 202 imagens, diferenca nula nao justifica a epoca mais tardia."""
    historico = [resultado(Split.VAL, 10), resultado(Split.VAL, 10)]
    assert best_epoch(historico) == 1


def test_melhor_epoca_recusa_historico_de_teste():
    historico = [resultado(Split.TEST_PUBLIC, 5), resultado(Split.TEST_PUBLIC, 15)]
    with pytest.raises(SplitPermissionError, match="test_public"):
        best_epoch(historico)


def test_melhor_epoca_recusa_historico_misturado():
    historico = [resultado(Split.VAL, 5), resultado(Split.TEST_PUBLIC, 20)]
    with pytest.raises(SplitPermissionError):
        best_epoch(historico)


def test_melhor_candidato_recusa_teste():
    # escolher tau da Trilha B no test_public e exatamente o que o enunciado proibe
    candidatos = {
        0.8: resultado(Split.TEST_PUBLIC, 10),
        0.95: resultado(Split.TEST_PUBLIC, 15),
    }
    with pytest.raises(SplitPermissionError):
        best_candidate(candidatos)


def test_melhor_candidato_no_val_funciona():
    candidatos = {0.8: resultado(Split.VAL, 10), 0.95: resultado(Split.VAL, 15)}
    assert best_candidate(candidatos) == 0.95


def test_is_improvement_recusa_teste():
    with pytest.raises(SplitPermissionError):
        is_improvement(resultado(Split.TEST_PUBLIC, 20), 0.0)


def test_select_by_recusa_teste():
    with pytest.raises(SplitPermissionError):
        select_by({"a": resultado(Split.TEST_PUBLIC, 10)}, key=lambda r: r.macro_f1)


def test_toda_funcao_publica_de_selecao_recusa_o_test_public():
    """Contrato do modulo: nenhuma porta de selecao aceita numero de teste.

    Se alguem adicionar uma funcao de selecao nova sem chamar ensure_selectable,
    este teste falha.
    """
    import inspect

    from src.evaluation import selection

    teste = resultado(Split.TEST_PUBLIC, 10)
    chamadas = {
        "best_epoch": lambda: selection.best_epoch([teste]),
        "best_candidate": lambda: selection.best_candidate({"a": teste}),
        "is_improvement": lambda: selection.is_improvement(teste, 0.0),
        "select_by": lambda: selection.select_by(
            {"a": teste}, key=lambda r: r.macro_f1
        ),
    }
    publicas = {
        nome
        for nome, objeto in inspect.getmembers(selection, inspect.isfunction)
        if not nome.startswith("_") and objeto.__module__ == selection.__name__
    }
    assert publicas == set(chamadas), (
        "funcao de selecao sem cobertura da barreira: "
        f"{sorted(publicas ^ set(chamadas))}"
    )
    for nome, chamada in chamadas.items():
        with pytest.raises(SplitPermissionError):
            chamada()


def test_metrica_desconhecida_tem_erro_claro():
    with pytest.raises(KeyError, match="metrica desconhecida"):
        resultado(Split.VAL, 10).metric("iou")
