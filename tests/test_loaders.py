"""Montagem dos DataLoader: de onde vem o rótulo de cada conjunto."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config.schema import DataConfig
from src.data.catalog import DataCatalog
from src.data.loaders import describe, labeled_loader, unlabeled_loader
from src.data.splits import Split


def para_tensor(imagem):
    """Transformação mínima para o collate funcionar: PIL -> tensor fixo."""
    pequena = np.asarray(imagem.resize((8, 8)), dtype=np.float32) / 255.0
    return torch.from_numpy(pequena).permute(2, 0, 1)


@pytest.fixture(scope="module")
def catalog(data_root) -> DataCatalog:
    return DataCatalog.open(data_root)


@pytest.fixture
def cfg(data_root) -> DataConfig:
    return DataConfig(root=data_root, batch_size=4, num_workers=0)


def test_rotulo_do_train_vem_do_orcamento(catalog, cfg):
    """No train só as imagens do orçamento têm rótulo; não há CSV do conjunto."""
    loader = labeled_loader(
        catalog, Split.TRAIN, cfg, transform=para_tensor, shuffle=False
    )
    assert len(loader.dataset) == 105  # orçamento padrão: 5 x 21


def test_orcamento_explicito_sobrepoe_o_da_configuracao(catalog, cfg):
    loader = labeled_loader(
        catalog, Split.TRAIN, cfg, transform=para_tensor, shuffle=False, budget=1
    )
    assert len(loader.dataset) == 21


@pytest.mark.parametrize("split,esperado", [(Split.VAL, 202), (Split.TEST_PUBLIC, 811)])
def test_rotulo_de_val_e_teste_vem_do_csv_do_conjunto(catalog, cfg, split, esperado):
    loader = labeled_loader(catalog, split, cfg, transform=para_tensor, shuffle=False)
    assert len(loader.dataset) == esperado


def test_batch_rotulado_tem_a_forma_esperada(catalog, cfg):
    loader = labeled_loader(
        catalog, Split.VAL, cfg, transform=para_tensor, shuffle=False
    )
    imagens, rotulos = next(iter(loader))
    assert imagens.shape == (4, 3, 8, 8)
    assert rotulos.shape == (4,)
    assert rotulos.dtype == torch.int64


def test_avaliacao_nao_descarta_o_ultimo_batch_incompleto(catalog, cfg):
    """Descartar batch na avaliação mudaria o denominador da métrica."""
    loader = labeled_loader(
        catalog, Split.VAL, cfg, transform=para_tensor, shuffle=False
    )
    assert loader.drop_last is False
    assert sum(len(y) for _, y in loader) == 202


def test_loader_sem_rotulo_entrega_o_pool_do_train(catalog, cfg):
    loader = unlabeled_loader(
        catalog,
        cfg,
        transform=para_tensor,
        image_ids=catalog.image_ids(Split.TRAIN)[:9],
    )
    imagens, ids = next(iter(loader))
    assert imagens.shape[0] == 4
    assert len(ids) == 4
    assert all(isinstance(i, str) for i in ids)


def test_describe_resume_as_fontes(catalog, cfg):
    loaders = {
        "labeled": labeled_loader(
            catalog, Split.TRAIN, cfg, transform=para_tensor, shuffle=False
        )
    }
    assert describe(loaders) == "labeled: 105 imagens"
