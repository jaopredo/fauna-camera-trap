"""Trilha B — semi-supervisionado com FixMatch.

Importa os modulos que registram coisas, para que o decorador rode quando
``src.tracks`` descobre esta pasta. O resto da trilha (``pseudo_labels``,
``figures``) e importado por quem precisa, sem efeito colateral.
"""

from src.tracks.b import strategy, transforms  # noqa: F401

__all__ = ["strategy", "transforms"]
