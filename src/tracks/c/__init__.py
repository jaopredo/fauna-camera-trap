"""Trilha C — professor e aluno.

Um foundation model de pesos abertos (BioCLIP, CLIP) rotula o ``train``, e uma
ResNet-18 aprende desses pseudo-rotulos junto com os 105 rotulos verdadeiros.

Importa ``strategy`` para que ``@register_strategy("distillation")`` rode quando
``src.tracks`` descobre esta pasta. O ``teacher`` nao e importado aqui de proposito:
ele depende de ``open_clip``, que e instalado a parte (``make setup-professor``), e
treinar o aluno nao precisa dele — so do arquivo que ele gravou.
"""

from src.tracks.c import strategy  # noqa: F401

__all__ = ["strategy"]
