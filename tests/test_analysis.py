"""Formato dos entregaveis: curva de orcamento e figuras."""

from __future__ import annotations

import csv

import pytest

from src.analysis.budget import (
    BUDGET_COLUMNS,
    BudgetRow,
    read_budget_curve,
    summarize,
    update_budget_curve,
    write_budget_curve,
)
from src.analysis.figures import plot_confusion_matrix, plot_training_curve

LINHAS = (
    BudgetRow("piso_finetune", 5, 0, 0.41, 0.52),
    BudgetRow("piso_finetune", 5, 42, 0.43, 0.54),
    BudgetRow("piso_scratch", 5, 0, 0.06, 0.09),
)


def test_csv_tem_as_colunas_do_enunciado(tmp_path):
    """metodo, orcamento, seed, F1 macro e acuracia — nesta ordem."""
    caminho = write_budget_curve(tmp_path / "curva_orcamento.csv", LINHAS)
    with caminho.open(newline="", encoding="utf-8") as handle:
        leitor = csv.reader(handle)
        assert tuple(next(leitor)) == BUDGET_COLUMNS
        assert tuple(next(leitor)) == ("piso_finetune", "5", "0", "0.41", "0.52")


def test_ida_e_volta_preserva_as_linhas(tmp_path):
    caminho = write_budget_curve(tmp_path / "c.csv", LINHAS)
    assert set(read_budget_curve(caminho)) == set(LINHAS)


def test_arquivo_ausente_le_como_vazio(tmp_path):
    assert read_budget_curve(tmp_path / "nao_existe.csv") == ()


def test_reexecutar_a_mesma_seed_substitui_em_vez_de_duplicar(tmp_path):
    """O CSV e gerado das execucoes; rodar o relatorio duas vezes nao pode duplicar."""
    caminho = tmp_path / "c.csv"
    update_budget_curve(caminho, LINHAS)
    update_budget_curve(caminho, [BudgetRow("piso_finetune", 5, 0, 0.99, 0.99)])

    linhas = read_budget_curve(caminho)
    assert len(linhas) == len(LINHAS)
    atualizada = next(r for r in linhas if r.key == ("piso_finetune", 5, 0))
    assert atualizada.f1_macro == pytest.approx(0.99)


def test_acrescentar_outro_orcamento_preserva_o_anterior(tmp_path):
    caminho = tmp_path / "c.csv"
    update_budget_curve(caminho, LINHAS)
    update_budget_curve(caminho, [BudgetRow("piso_finetune", 20, 0, 0.55, 0.61)])
    assert len(read_budget_curve(caminho)) == len(LINHAS) + 1


def test_resumo_traz_media_e_desvio_por_metodo():
    resumos = {r.metodo: r for r in summarize(LINHAS)}
    finetune = resumos["piso_finetune"]
    assert finetune.seeds == 2
    assert finetune.f1_medio == pytest.approx(0.42)
    assert "±" in finetune.f1_formatado()


def test_uma_seed_nao_reporta_desvio():
    """Desvio zero sugeriria estabilidade que nao foi medida."""
    resumo = {r.metodo: r for r in summarize(LINHAS)}["piso_scratch"]
    assert resumo.seeds == 1
    assert resumo.f1_formatado() == "0.060"
    assert "±" not in resumo.f1_formatado()


def test_curva_de_treino_grava_arquivo(tmp_path):
    historico = [
        {
            "epoch": e,
            "train_loss": 3.0 / e,
            "val_macro_f1": 0.1 * e,
            "val_accuracy": 0.2,
        }
        for e in range(1, 6)
    ]
    caminho = plot_training_curve(historico, tmp_path / "fig" / "curva.png")
    assert caminho.exists() and caminho.stat().st_size > 1000


def test_curva_funciona_sem_metrica_de_validacao(tmp_path):
    historico = [{"epoch": 1, "train_loss": 2.0}, {"epoch": 2, "train_loss": 1.0}]
    assert plot_training_curve(historico, tmp_path / "c.png").exists()


def test_matriz_de_confusao_grava_arquivo(tmp_path):
    confusao = [[5, 1, 0], [0, 4, 2], [1, 0, 3]]
    caminho = plot_confusion_matrix(confusao, ["a", "b", "c"], tmp_path / "m.png")
    assert caminho.exists() and caminho.stat().st_size > 1000


def test_matriz_com_classe_sem_suporte_nao_divide_por_zero(tmp_path):
    """Cavalo tem 4 imagens no test_public; uma classe pode ficar sem nenhuma."""
    confusao = [[2, 0, 0], [0, 0, 0], [0, 0, 1]]
    assert plot_confusion_matrix(confusao, ["a", "b", "c"], tmp_path / "m.png").exists()
