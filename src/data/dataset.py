"""Datasets de imagem do ENA24, rotulado e nao rotulado.

Duas classes porque os dois casos tem contratos diferentes de verdade: o rotulado
devolve ``(imagem, rotulo)``; o nao rotulado devolve ``(imagem, image_id)``, e o id
e o que permite guardar pseudo-rotulo por imagem (trilhas B e C).

Nao herdam de ``torch.utils.data.Dataset``: o ``DataLoader`` aceita qualquer objeto
com ``__len__`` e ``__getitem__``, e assim este modulo nao importa PyTorch — o que
mantem a leitura de dados testavel sem ele.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image

from src.data.catalog import DataCatalog, LabeledExample
from src.data.splits import Split

#: Uma transformacao recebe a imagem PIL e devolve o que o modelo consome
#: (tipicamente um tensor). ``None`` devolve a propria imagem PIL.
Transform = Callable[[Image.Image], Any]


def load_rgb(path: Path) -> Image.Image:
    """Abre a imagem em RGB.

    O ``convert`` nao e opcional: boa parte do pacote e noturna em infravermelho e
    vem em tons de cinza, com um canal so. Sem a conversao, o batch misturaria
    imagens de 1 e 3 canais e o ``collate`` quebraria.
    """
    with Image.open(path) as image:
        return image.convert("RGB")


@dataclass(frozen=True)
class LabeledImageDataset:
    """Imagens com rotulo: orcamento do train, val ou test_public."""

    paths: tuple[Path, ...]
    labels: tuple[int, ...]
    transform: Transform | None = None

    def __post_init__(self) -> None:
        if len(self.paths) != len(self.labels):
            raise ValueError(
                f"paths e labels com tamanhos diferentes: "
                f"{len(self.paths)} != {len(self.labels)}"
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        image = load_rgb(self.paths[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    @classmethod
    def from_examples(
        cls,
        catalog: DataCatalog,
        split: Split,
        examples: Sequence[LabeledExample],
        *,
        transform: Transform | None = None,
    ) -> "LabeledImageDataset":
        """Monta o dataset resolvendo os caminhos em ``split``.

        ``split`` e explicito porque os rotulos do orcamento apontam para imagens
        que vivem em ``images/train``, nao numa pasta propria.
        """
        return cls(
            paths=tuple(catalog.image_path(split, e.image_id) for e in examples),
            labels=tuple(e.label_index for e in examples),
            transform=transform,
        )


@dataclass(frozen=True)
class UnlabeledImageDataset:
    """O pool sem rotulo. Devolve ``(imagem, image_id)``.

    O id acompanha a imagem para que pseudo-rotulos possam ser gravados e auditados
    por imagem, em vez de depender da ordem do loader.
    """

    paths: tuple[Path, ...]
    image_ids: tuple[str, ...]
    transform: Transform | None = None

    def __post_init__(self) -> None:
        if len(self.paths) != len(self.image_ids):
            raise ValueError("paths e image_ids com tamanhos diferentes")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Any, str]:
        image = load_rgb(self.paths[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.image_ids[index]

    @classmethod
    def from_catalog(
        cls,
        catalog: DataCatalog,
        split: Split = Split.TRAIN,
        *,
        image_ids: Sequence[str] | None = None,
        transform: Transform | None = None,
    ) -> "UnlabeledImageDataset":
        """Todas as imagens de ``split``, ou apenas ``image_ids`` se informado.

        O padrao e o pool inteiro do ``train``, inclusive as imagens do orcamento:
        e assim que o enunciado descreve o FixMatch ("o train inteiro como nao
        rotulado"). Filtrar as rotuladas, quando for o caso, e tarefa do metodo.
        """
        ids = tuple(image_ids) if image_ids is not None else catalog.image_ids(split)
        return cls(
            paths=tuple(catalog.image_path(split, i) for i in ids),
            image_ids=ids,
            transform=transform,
        )
