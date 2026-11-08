"""Leitura das imagens do pacote."""

from __future__ import annotations

import pytest

from src.data.catalog import DataCatalog
from src.data.dataset import LabeledImageDataset, UnlabeledImageDataset, load_rgb
from src.data.splits import Split


@pytest.fixture(scope="module")
def catalog(data_root) -> DataCatalog:
    return DataCatalog.open(data_root)


def test_imagem_sempre_chega_com_tres_canais(catalog):
    """Boa parte do pacote e noturna em infravermelho, em tons de cinza.

    Sem a conversao para RGB, o batch misturaria imagens de 1 e 3 canais e o
    collate do DataLoader quebraria no meio do treino.
    """
    for exemplo in catalog.labeled(Split.VAL)[:12]:
        imagem = load_rgb(catalog.image_path(Split.VAL, exemplo.image_id))
        assert imagem.mode == "RGB"


def test_proporcoes_variam_entre_imagens(catalog):
    """O corte da faixa de texto deixou proporcoes diferentes por camera.

    Por isso a politica de aumentacao tem de redimensionar: nao se pode supor
    um tamanho fixo vindo do disco.
    """
    tamanhos = {
        load_rgb(catalog.image_path(Split.VAL, e.image_id)).size
        for e in catalog.labeled(Split.VAL)[:30]
    }
    assert len(tamanhos) > 1
    assert all(max(w, h) <= 512 for w, h in tamanhos)


def test_dataset_rotulado_devolve_imagem_e_rotulo(catalog):
    exemplos = catalog.budget(1)
    dataset = LabeledImageDataset.from_examples(catalog, Split.TRAIN, exemplos)
    assert len(dataset) == len(exemplos) == 21
    imagem, rotulo = dataset[0]
    assert imagem.mode == "RGB"
    assert rotulo == exemplos[0].label_index


def test_dataset_rotulado_aplica_a_transformacao(catalog):
    dataset = LabeledImageDataset.from_examples(
        catalog, Split.TRAIN, catalog.budget(1), transform=lambda img: img.size
    )
    tamanho, _ = dataset[0]
    assert isinstance(tamanho, tuple)


def test_dataset_sem_rotulo_devolve_o_id(catalog):
    """O id acompanha a imagem para que pseudo-rotulo possa ser auditado."""
    ids = catalog.image_ids(Split.TRAIN)[:5]
    dataset = UnlabeledImageDataset.from_catalog(catalog, image_ids=ids)
    imagem, image_id = dataset[2]
    assert image_id == ids[2]
    assert imagem.mode == "RGB"


def test_pool_sem_rotulo_e_o_train_inteiro(catalog):
    """O enunciado manda usar "o train inteiro como nao rotulado"."""
    dataset = UnlabeledImageDataset.from_catalog(catalog)
    assert len(dataset) == 6505
    rotuladas = {e.image_id for e in catalog.budget(5)}
    assert rotuladas <= set(dataset.image_ids)


def test_tamanhos_incompativeis_sao_recusados():
    with pytest.raises(ValueError, match="tamanhos diferentes"):
        LabeledImageDataset(paths=(), labels=(1,))
