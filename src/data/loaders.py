"""Montagem dos ``DataLoader`` a partir da configuracao.

Unico modulo de ``data`` que importa PyTorch: e a borda entre a leitura de dados
(pura) e o treino. Os metodos declaram de quais fontes precisam
(``TrainingStrategy.required_sources``) e recebem aqui o dicionario correspondente.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from torch.utils.data import DataLoader

from src.config.schema import DataConfig
from src.data.catalog import DataCatalog
from src.data.dataset import (
    LabeledImageDataset,
    Transform,
    UnlabeledImageDataset,
)
from src.data.splits import Split

#: Nomes das fontes de dados que um metodo de treino pode pedir.
LABELED = "labeled"
UNLABELED = "unlabeled"


def labeled_loader(
    catalog: DataCatalog,
    split: Split,
    cfg: DataConfig,
    *,
    transform: Transform | None,
    shuffle: bool,
    budget: int | None = None,
) -> DataLoader:
    """Loader de um conjunto rotulado.

    Para ``Split.TRAIN`` os rotulos vem do orcamento (``budget``, ou
    ``cfg.budget``); para ``val`` e ``test_public``, do CSV do proprio conjunto.
    """
    if split is Split.TRAIN:
        examples = catalog.budget(budget if budget is not None else cfg.budget)
    else:
        examples = catalog.labeled(split)
    dataset = LabeledImageDataset.from_examples(
        catalog, split, examples, transform=transform
    )
    return _loader(dataset, cfg, shuffle=shuffle, drop_last=shuffle)


def unlabeled_loader(
    catalog: DataCatalog,
    cfg: DataConfig,
    *,
    transform: Transform | None,
    image_ids: Iterable[str] | None = None,
) -> DataLoader:
    """Loader do pool sem rotulo (as 6.505 imagens de ``train``)."""
    dataset = UnlabeledImageDataset.from_catalog(
        catalog,
        Split.TRAIN,
        image_ids=None if image_ids is None else tuple(image_ids),
        transform=transform,
    )
    return _loader(dataset, cfg, shuffle=True, drop_last=True)


def _loader(dataset, cfg: DataConfig, *, shuffle: bool, drop_last: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        drop_last=drop_last,
        # pin_memory so ajuda com GPU; o modelo entregue roda em CPU
        pin_memory=False,
        persistent_workers=cfg.num_workers > 0,
    )


def describe(loaders: Mapping[str, DataLoader]) -> str:
    """Resumo de uma linha por fonte — vai para o log da execucao."""
    return " | ".join(
        f"{name}: {len(loader.dataset)} imagens"
        for name, loader in sorted(loaders.items())
    )
