"""Conjuntos do pacote de dados e a barreira que protege o ``test_public``.

O enunciado separa os conjuntos por **permissao**, nao por computacao: no ``val`` se
escolhe epoca, limiar e arquitetura; o ``test_public`` so pode ser reportado. Como a
computacao e identica nos dois, a diferenca vive aqui, num unico lugar, em vez de em
dois modulos de avaliacao duplicados.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

from src.errors import AppError


class Split(str, Enum):
    """Conjuntos presentes no pacote ``pa3_dados``.

    ``test_private`` nao aparece: ele fica com a monitoria e so e visto pelo
    ``predict.py``, que recebe uma pasta de imagens sem rotulo algum.
    """

    TRAIN = "train"
    VAL = "val"
    TEST_PUBLIC = "test_public"


#: Conjuntos com rotulo humano completo. ``train`` fica de fora: so as imagens
#: listadas nos arquivos de orcamento tem rotulo.
LABELED_SPLITS: Final = frozenset({Split.VAL, Split.TEST_PUBLIC})

#: O unico conjunto em que selecao de modelo e permitida.
SELECTION_SPLITS: Final = frozenset({Split.VAL})

#: Orcamentos de rotulo do pacote, aninhados: 1 esta contido em 5, que esta em 20.
BUDGETS: Final = (1, 5, 20)

#: Orcamento padrao do PA; os outros dois servem a curva da Parte 2.
DEFAULT_BUDGET: Final = 5


class SplitPermissionError(AppError, RuntimeError):
    """Tentativa de usar um conjunto proibido para selecao de modelo."""


class UnknownSplitError(AppError, ValueError):
    """Nome de conjunto que nao existe no pacote de dados."""


class InvalidBudgetError(AppError, ValueError):
    """Orcamento de rotulos que nao existe no pacote de dados."""


def parse_split(name: str | Split) -> Split:
    """Converte texto em ``Split``, recusando nomes desconhecidos.

    Falhar aqui e barato: um ``--split teste_publico`` que passasse em silencio
    levaria a avaliar o conjunto errado e a reportar um numero invalido.
    """
    if isinstance(name, Split):
        return name
    try:
        return Split(name)
    except ValueError:
        raise UnknownSplitError(
            f"conjunto desconhecido: {name!r}. Disponiveis: "
            f"{sorted(s.value for s in Split)}"
        ) from None


def ensure_selectable(split: Split) -> Split:
    """Autoriza o uso de ``split`` para escolher epoca, limiar ou arquitetura.

    Levanta ``SplitPermissionError`` para qualquer conjunto fora de
    ``SELECTION_SPLITS``. Toda funcao de ``src.evaluation.selection`` passa por
    aqui, de modo que nao existe caminho de selecao que aceite o ``test_public``.
    """
    if split not in SELECTION_SPLITS:
        raise SplitPermissionError(
            f"selecao de modelo em {split.value!r} e proibida pelo enunciado "
            f"(secao 3: treinar ou fazer selecao de modelo no test_public). "
            f"Use {sorted(s.value for s in SELECTION_SPLITS)}."
        )
    return split


def validate_budget(labels_per_class: int) -> int:
    """Confere que o orcamento e um dos tres distribuidos no pacote."""
    if labels_per_class not in BUDGETS:
        raise InvalidBudgetError(
            f"orcamento invalido: {labels_per_class!r}. Disponiveis: {list(BUDGETS)}"
        )
    return labels_per_class
