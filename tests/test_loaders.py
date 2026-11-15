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


# ===================== mu do FixMatch =======================================


def test_lote_sem_rotulo_e_mu_vezes_o_rotulado(catalog, data_root):
    """A aula 13 (p. 87) lista a razao mu entre os dois lotes como entrada."""
    from dataclasses import replace

    cfg = DataConfig(
        root=data_root, batch_size=4, num_workers=0, unlabeled_batch_ratio=7
    )
    assert cfg.unlabeled_batch_size == 28

    loader = unlabeled_loader(
        catalog,
        cfg,
        transform=para_tensor,
        image_ids=catalog.image_ids(Split.TRAIN)[:60],
    )
    imagens, _ids = next(iter(loader))
    assert imagens.shape[0] == 28

    sem_razao = replace(cfg, unlabeled_batch_ratio=1)
    outro = unlabeled_loader(
        catalog,
        sem_razao,
        transform=para_tensor,
        image_ids=catalog.image_ids(Split.TRAIN)[:60],
    )
    assert next(iter(outro))[0].shape[0] == 4


def test_loader_sem_rotulo_descarta_lote_incompleto(catalog, data_root):
    """O FixMatch divide a perda por mu*B; um ultimo lote curto mudaria o peso
    relativo dos dois termos no fim de cada epoca."""
    cfg = DataConfig(
        root=data_root, batch_size=4, num_workers=0, unlabeled_batch_ratio=7
    )
    loader = unlabeled_loader(
        catalog,
        cfg,
        transform=para_tensor,
        image_ids=catalog.image_ids(Split.TRAIN)[:30],
    )
    assert loader.drop_last is True
    assert all(imagens.shape[0] == 28 for imagens, _ in loader)
