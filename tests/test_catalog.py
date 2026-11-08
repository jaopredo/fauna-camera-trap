"""Leitura do pacote de dados e o aninhamento dos orcamentos."""

from __future__ import annotations

import pytest

from src.data.catalog import CatalogError, ClassVocabulary, DataCatalog
from src.data.splits import BUDGETS, Split

ESPERADO_POR_SPLIT = {Split.TRAIN: 6505, Split.VAL: 202, Split.TEST_PUBLIC: 811}


@pytest.fixture(scope="module")
def catalog(data_root) -> DataCatalog:
    return DataCatalog.open(data_root)


def test_vinte_e_uma_classes_na_ordem_do_arquivo(catalog):
    assert len(catalog.classes) == 21
    assert catalog.classes.labels[0] == "american_black_bear"
    assert catalog.classes.index("horse") == 13
    assert catalog.classes.label(13) == "horse"


@pytest.mark.parametrize("split,esperado", ESPERADO_POR_SPLIT.items())
def test_contagem_de_imagens_por_split(catalog, split, esperado):
    assert len(catalog.image_ids(split)) == esperado


@pytest.mark.parametrize("budget", BUDGETS)
def test_orcamento_tem_k_rotulos_por_classe(catalog, budget):
    exemplos = catalog.budget(budget)
    assert len(exemplos) == budget * len(catalog.classes)
    por_classe: dict[int, int] = {}
    for exemplo in exemplos:
        por_classe[exemplo.label_index] = por_classe.get(exemplo.label_index, 0) + 1
    assert set(por_classe) == set(range(len(catalog.classes)))
    assert set(por_classe.values()) == {budget}


def test_orcamentos_sao_aninhados(catalog):
    """1 contido em 5 contido em 20 — o enunciado garante, o teste confere.

    Se isso quebrar, a curva da Parte 2 deixa de medir o efeito de mais rotulos e
    passa a medir tambem a troca do conjunto rotulado.
    """
    ids = {b: {e.image_id for e in catalog.budget(b)} for b in BUDGETS}
    assert ids[1] < ids[5] < ids[20]


def test_rotulos_do_orcamento_apontam_para_imagens_do_train(catalog):
    treino = set(catalog.image_ids(Split.TRAIN))
    for budget in BUDGETS:
        ids = {e.image_id for e in catalog.budget(budget)}
        assert ids <= treino


def test_val_e_test_public_nao_se_sobrepoem(catalog):
    val = {e.image_id for e in catalog.labeled(Split.VAL)}
    teste = {e.image_id for e in catalog.labeled(Split.TEST_PUBLIC)}
    assert not (val & teste)


@pytest.mark.parametrize("split", [Split.VAL, Split.TEST_PUBLIC])
def test_csv_de_rotulos_cobre_as_imagens_da_pasta(catalog, split):
    rotulados = {e.image_id for e in catalog.labeled(split)}
    assert rotulados == set(catalog.image_ids(split))


def test_caminho_de_imagem_existe(catalog):
    exemplo = catalog.budget(1)[0]
    assert catalog.image_path(Split.TRAIN, exemplo.image_id).exists()


def test_train_nao_tem_csv_de_rotulos_proprio(catalog):
    with pytest.raises(CatalogError, match="orcamento"):
        catalog.labels_csv(Split.TRAIN)


def test_pacote_ausente_tem_erro_claro(tmp_path):
    with pytest.raises(CatalogError, match="pa3_dados.zip"):
        DataCatalog.open(tmp_path)


def test_vocabulario_recusa_indices_fora_de_ordem(tmp_path):
    path = tmp_path / "classes.csv"
    path.write_text("idx,label\n0,a\n2,b\n", encoding="utf-8")
    with pytest.raises(CatalogError, match="0..N-1"):
        ClassVocabulary.from_csv(path)


def test_vocabulario_recusa_rotulo_fora_do_arquivo():
    vocabulario = ClassVocabulary(("a", "b"))
    with pytest.raises(CatalogError, match="fora de classes.csv"):
        vocabulario.index("c")
