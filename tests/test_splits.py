"""A barreira do test_public e a validacao de orcamentos."""

from __future__ import annotations

import pytest

from src.data.splits import (
    BUDGETS,
    SELECTION_SPLITS,
    InvalidBudgetError,
    Split,
    SplitPermissionError,
    UnknownSplitError,
    ensure_selectable,
    parse_split,
    validate_budget,
)


def test_val_e_o_unico_conjunto_de_selecao():
    assert SELECTION_SPLITS == {Split.VAL}


def test_selecao_no_val_e_permitida():
    assert ensure_selectable(Split.VAL) is Split.VAL


@pytest.mark.parametrize("split", [Split.TEST_PUBLIC, Split.TRAIN])
def test_selecao_fora_do_val_e_recusada(split):
    with pytest.raises(SplitPermissionError, match="proibida pelo enunciado"):
        ensure_selectable(split)


def test_parse_split_aceita_nome_e_enum():
    assert parse_split("test_public") is Split.TEST_PUBLIC
    assert parse_split(Split.VAL) is Split.VAL


def test_parse_split_recusa_nome_desconhecido():
    with pytest.raises(UnknownSplitError, match="Disponiveis"):
        parse_split("teste_publico")


def test_test_private_nao_e_um_split_do_pacote():
    # o conjunto privado fica com a monitoria: so o predict.py o ve, como pasta
    with pytest.raises(UnknownSplitError):
        parse_split("test_private")


@pytest.mark.parametrize("budget", BUDGETS)
def test_orcamentos_do_pacote_sao_validos(budget):
    assert validate_budget(budget) == budget


@pytest.mark.parametrize("budget", [0, 2, 10, 100])
def test_orcamento_fora_do_pacote_e_recusado(budget):
    with pytest.raises(InvalidBudgetError):
        validate_budget(budget)
