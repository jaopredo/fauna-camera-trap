"""Leitura do pacote de dados: classes, orcamentos de rotulo e caminhos de imagem.

Responsabilidade unica: traduzir o que esta em disco (``classes.csv``, os
``rotulos_K.csv``, as pastas de imagem) em objetos tipados. Nao abre imagem, nao
aplica transformacao e nao conhece PyTorch.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from src.data.splits import LABELED_SPLITS, Split, validate_budget
from src.errors import AppError

#: Extensao das imagens do pacote. O ``image_id`` do CSV e o nome do arquivo sem ela.
IMAGE_SUFFIX = ".jpg"


class CatalogError(AppError, RuntimeError):
    """Pacote de dados ausente, incompleto ou inconsistente."""


@dataclass(frozen=True)
class ClassVocabulary:
    """As classes do problema, na ordem de ``classes.csv``.

    A ordem e o indice vem do arquivo, nunca de uma lista escrita no codigo: o
    ``predict.py`` precisa emitir exatamente os nomes de ``classes.csv``.
    """

    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.labels)) != len(self.labels):
            raise CatalogError("classes.csv tem rotulos repetidos")

    def __len__(self) -> int:
        return len(self.labels)

    @cached_property
    def _index_by_label(self) -> dict[str, int]:
        return {label: i for i, label in enumerate(self.labels)}

    def index(self, label: str) -> int:
        try:
            return self._index_by_label[label]
        except KeyError:
            raise CatalogError(f"rotulo fora de classes.csv: {label!r}") from None

    def label(self, index: int) -> str:
        try:
            return self.labels[index]
        except IndexError:
            raise CatalogError(
                f"indice de classe fora da faixa: {index!r} (0..{len(self) - 1})"
            ) from None

    @classmethod
    def from_csv(cls, path: Path) -> "ClassVocabulary":
        """Le ``classes.csv`` (colunas ``idx,label``), exigindo indices 0..N-1 em ordem."""
        rows = _read_csv(path, required={"idx", "label"})
        labels: list[str] = []
        for expected, row in enumerate(rows):
            idx = int(row["idx"])
            if idx != expected:
                raise CatalogError(
                    f"{path}: indices precisam ser 0..N-1 em ordem; "
                    f"esperava {expected}, veio {idx}"
                )
            labels.append(row["label"])
        if not labels:
            raise CatalogError(f"{path}: nenhuma classe encontrada")
        return cls(tuple(labels))


@dataclass(frozen=True)
class LabeledExample:
    """Uma imagem com rotulo humano."""

    image_id: str
    label_index: int


@dataclass(frozen=True)
class DataCatalog:
    """Indice do pacote de dados sob ``root``.

    Estrutura esperada (ver ``data/LEIAME.md``)::

        root/classes.csv
        root/labels/{val,test_public,rotulos_1,rotulos_5,rotulos_20}.csv
        root/images/{train,val,test_public}/<image_id>.jpg
    """

    root: Path
    classes: ClassVocabulary

    @classmethod
    def open(cls, root: Path | str) -> "DataCatalog":
        root = Path(root)
        classes_csv = root / "classes.csv"
        if not classes_csv.exists():
            raise CatalogError(
                f"pacote de dados nao encontrado: {classes_csv} nao existe. "
                f"Descompacte pa3_dados.zip em {root}."
            )
        return cls(root=root, classes=ClassVocabulary.from_csv(classes_csv))

    # ---------------------------------------------------------------- caminhos
    def images_dir(self, split: Split) -> Path:
        return self.root / "images" / split.value

    def image_path(self, split: Split, image_id: str) -> Path:
        return self.images_dir(split) / f"{image_id}{IMAGE_SUFFIX}"

    def labels_csv(self, split: Split) -> Path:
        if split not in LABELED_SPLITS:
            raise CatalogError(
                f"{split.value} nao tem arquivo de rotulos proprio; os rotulos do "
                f"train vem dos orcamentos (ver budget())"
            )
        return self.root / "labels" / f"{split.value}.csv"

    def budget_csv(self, labels_per_class: int) -> Path:
        return self.root / "labels" / f"rotulos_{validate_budget(labels_per_class)}.csv"

    # ------------------------------------------------------------------ listas
    def image_ids(self, split: Split) -> tuple[str, ...]:
        """Todos os ids presentes na pasta de imagens do conjunto, ordenados.

        Para o ``train`` isso e o pool inteiro (6.505 imagens), rotulado ou nao —
        e o que as trilhas A e B consomem como dado sem rotulo.
        """
        directory = self.images_dir(split)
        if not directory.is_dir():
            raise CatalogError(f"pasta de imagens ausente: {directory}")
        return tuple(sorted(p.stem for p in directory.glob(f"*{IMAGE_SUFFIX}")))

    def labeled(self, split: Split) -> tuple[LabeledExample, ...]:
        """Exemplos rotulados de ``val`` ou ``test_public``.

        Nao ha barreira aqui: a Parte 0 precisa reportar numeros no ``test_public``
        e a Trilha B precisa rodar diagnostico nele. A barreira fica na selecao
        de modelo (``src.data.splits.ensure_selectable``).
        """
        return self._read_labeled(self.labels_csv(split))

    def budget(self, labels_per_class: int) -> tuple[LabeledExample, ...]:
        """Os rotulos do orcamento ``labels_per_class`` — subconjunto do ``train``."""
        return self._read_labeled(self.budget_csv(labels_per_class))

    def _read_labeled(self, path: Path) -> tuple[LabeledExample, ...]:
        rows = _read_csv(path, required={"image_id", "label"})
        return tuple(
            LabeledExample(
                image_id=row["image_id"],
                label_index=self.classes.index(row["label"]),
            )
            for row in rows
        )


def _read_csv(path: Path, *, required: set[str]) -> list[dict[str, str]]:
    """Le um CSV com cabecalho, exigindo as colunas de ``required``."""
    if not path.exists():
        raise CatalogError(f"arquivo nao encontrado: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise CatalogError(
                f"{path}: colunas ausentes {sorted(missing)}; "
                f"encontradas {list(reader.fieldnames or ())}"
            )
        return [row for row in reader]
