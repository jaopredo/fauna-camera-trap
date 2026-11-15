"""Reabrir execucoes salvas: a metrica de validacao sem carregar modelo.

O foco aqui e `best_validation_metric`, que os relatorios usam para decidir **qual
execucao reportar**. Essa decisao tem de sair do `val`; se saisse do `test_public`
seria selecao de modelo no conjunto de reporte, que o enunciado proibe.
"""

from __future__ import annotations

import json

import pytest

from src.errors import AppError
from src.evaluation.runs import best_validation_metric


def escrever_historico(run_dir, epocas) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "history.json").write_text(json.dumps(epocas), encoding="utf-8")


def test_devolve_o_melhor_valor_e_a_epoca(tmp_path):
    escrever_historico(
        tmp_path / "exp",
        [
            {"epoch": 1, "val_macro_f1": 0.10},
            {"epoch": 2, "val_macro_f1": 0.42},
            {"epoch": 3, "val_macro_f1": 0.31},
        ],
    )
    assert best_validation_metric(tmp_path / "exp") == (0.42, 2)


def test_empate_nao_quebra(tmp_path):
    escrever_historico(
        tmp_path / "exp",
        [{"epoch": 1, "val_macro_f1": 0.5}, {"epoch": 2, "val_macro_f1": 0.5}],
    )
    f1, epoca = best_validation_metric(tmp_path / "exp")
    assert f1 == pytest.approx(0.5)
    assert epoca in (1, 2)


def test_aceita_outra_metrica_monitorada(tmp_path):
    escrever_historico(
        tmp_path / "exp",
        [{"epoch": 1, "val_accuracy": 0.7}, {"epoch": 2, "val_accuracy": 0.9}],
    )
    assert best_validation_metric(tmp_path / "exp", monitor="accuracy") == (0.9, 2)


def test_ignora_metrica_que_nao_seja_de_validacao(tmp_path):
    """A garantia que importa: a funcao le `val_<monitor>` e **so** isso.

    Mesmo que alguem grave uma metrica de teste no historico — o que o laco de treino
    nunca faz, porque nunca ve o test_public —, ela nao e considerada. Sem isto, a
    escolha de qual execucao reportar poderia silenciosamente passar a vir do teste.
    """
    escrever_historico(
        tmp_path / "exp",
        [
            {"epoch": 1, "val_macro_f1": 0.30, "test_macro_f1": 0.99},
            {"epoch": 2, "val_macro_f1": 0.40, "test_macro_f1": 0.11},
        ],
    )
    f1, epoca = best_validation_metric(tmp_path / "exp")
    assert (f1, epoca) == (0.40, 2), "escolheu pela metrica errada"


def test_epoca_sem_a_metrica_e_pulada(tmp_path):
    """Retomada de treino pode deixar epoca sem avaliacao; nao pode virar erro."""
    escrever_historico(
        tmp_path / "exp",
        [
            {"epoch": 1, "val_macro_f1": 0.20},
            {"epoch": 2, "train_loss": 0.5},
            {"epoch": 3, "val_macro_f1": 0.35},
        ],
    )
    assert best_validation_metric(tmp_path / "exp") == (0.35, 3)


def test_sem_history_json_a_mensagem_diz_o_que_houve(tmp_path):
    (tmp_path / "exp").mkdir()
    with pytest.raises(AppError, match="history.json"):
        best_validation_metric(tmp_path / "exp")


def test_metrica_ausente_lista_as_chaves_disponiveis(tmp_path):
    escrever_historico(tmp_path / "exp", [{"epoch": 1, "train_loss": 0.5}])
    with pytest.raises(AppError, match="train_loss"):
        best_validation_metric(tmp_path / "exp")


def test_historico_vazio_nao_quebra_com_indice(tmp_path):
    escrever_historico(tmp_path / "exp", [])
    with pytest.raises(AppError, match="vazio"):
        best_validation_metric(tmp_path / "exp")


def test_aceita_historico_embrulhado_em_dict(tmp_path):
    """Formato antigo gravava `{"epochs": [...]}`; os dois precisam funcionar."""
    run_dir = tmp_path / "exp"
    run_dir.mkdir()
    (run_dir / "history.json").write_text(
        json.dumps({"epochs": [{"epoch": 1, "val_macro_f1": 0.5}]}), encoding="utf-8"
    )
    assert best_validation_metric(run_dir) == (0.5, 1)
