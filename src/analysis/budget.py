"""``curva_orcamento.csv``: os numeros da Parte 2, gerados das execucoes.

O arquivo e um entregavel e **nunca e editado a mao** — cada linha vem de uma
execucao que deixou checkpoint e manifesto em ``outputs/``. As colunas sao as que o
enunciado pede: metodo, orcamento, seed, F1 macro e acuracia.

O F1 e sempre o do ``test_public``: e o conjunto de reporte. A selecao de epoca que
produziu o checkpoint foi feita no ``val``, antes, por
``src.evaluation.selection``.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Iterable, Mapping, Sequence

#: Ordem exata das colunas do entregavel.
BUDGET_COLUMNS = ("metodo", "orcamento", "seed", "f1_macro", "acuracia")


@dataclass(frozen=True)
class BudgetRow:
    """Uma execucao: um metodo, num orcamento, com uma seed."""

    metodo: str
    orcamento: int
    seed: int
    f1_macro: float
    acuracia: float

    @property
    def key(self) -> tuple[str, int, int]:
        """Identidade da execucao: reexecutar a mesma combinacao substitui a linha."""
        return (self.metodo, self.orcamento, self.seed)

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "BudgetRow":
        return cls(
            metodo=str(row["metodo"]),
            orcamento=int(row["orcamento"]),
            seed=int(row["seed"]),
            f1_macro=float(row["f1_macro"]),
            acuracia=float(row["acuracia"]),
        )


def read_budget_curve(path: Path | str) -> tuple[BudgetRow, ...]:
    """Le o CSV, ou devolve vazio se ele ainda nao existe."""
    path = Path(path)
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(BudgetRow.from_mapping(row) for row in csv.DictReader(handle))


def merge_rows(
    existing: Iterable[BudgetRow], new: Iterable[BudgetRow]
) -> tuple[BudgetRow, ...]:
    """Funde linhas; em caso de mesma (metodo, orcamento, seed), a nova vence.

    E o que permite rodar ``report`` varias vezes, ou acrescentar um orcamento,
    sem duplicar linha nem perder as anteriores.
    """
    por_chave = {linha.key: linha for linha in existing}
    por_chave.update({linha.key: linha for linha in new})
    return tuple(sorted(por_chave.values(), key=lambda r: r.key))


def write_budget_curve(path: Path | str, rows: Sequence[BudgetRow]) -> Path:
    """Grava o CSV com as colunas na ordem do entregavel."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nomes = tuple(f.name for f in fields(BudgetRow))
    assert nomes == BUDGET_COLUMNS, "os campos precisam casar com BUDGET_COLUMNS"
    with path.open("w", newline="", encoding="utf-8") as handle:
        escritor = csv.writer(handle)
        escritor.writerow(BUDGET_COLUMNS)
        escritor.writerows(astuple(linha) for linha in rows)
    return path


def update_budget_curve(path: Path | str, rows: Sequence[BudgetRow]) -> Path:
    """Le, funde e regrava — a operacao que os scripts de relatorio usam."""
    return write_budget_curve(path, merge_rows(read_budget_curve(path), rows))


@dataclass(frozen=True)
class BudgetSummary:
    """Agregado de um metodo num orcamento, sobre as seeds disponiveis."""

    metodo: str
    orcamento: int
    seeds: int
    f1_medio: float
    f1_desvio: float
    acuracia_media: float
    acuracia_desvio: float

    def f1_formatado(self) -> str:
        """``0,412 ± 0,018`` — com uma seed so, sem o desvio."""
        if self.seeds < 2:
            return f"{self.f1_medio:.3f}"
        return f"{self.f1_medio:.3f} ± {self.f1_desvio:.3f}"

    def acuracia_formatada(self) -> str:
        if self.seeds < 2:
            return f"{self.acuracia_media:.3f}"
        return f"{self.acuracia_media:.3f} ± {self.acuracia_desvio:.3f}"


def summarize(rows: Iterable[BudgetRow]) -> tuple[BudgetSummary, ...]:
    """Media e desvio por (metodo, orcamento).

    Desvio amostral, e so quando ha ao menos duas seeds. Com uma seed o desvio nao
    existe — e reportar zero ali sugeriria uma estabilidade que nao foi medida.
    """
    grupos: dict[tuple[str, int], list[BudgetRow]] = {}
    for linha in rows:
        grupos.setdefault((linha.metodo, linha.orcamento), []).append(linha)

    resumos = []
    for (metodo, orcamento), grupo in sorted(grupos.items()):
        f1 = [linha.f1_macro for linha in grupo]
        acuracia = [linha.acuracia for linha in grupo]
        resumos.append(
            BudgetSummary(
                metodo=metodo,
                orcamento=orcamento,
                seeds=len(grupo),
                f1_medio=statistics.fmean(f1),
                f1_desvio=statistics.stdev(f1) if len(f1) > 1 else 0.0,
                acuracia_media=statistics.fmean(acuracia),
                acuracia_desvio=statistics.stdev(acuracia)
                if len(acuracia) > 1
                else 0.0,
            )
        )
    return tuple(resumos)
