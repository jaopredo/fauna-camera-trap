"""Coloca a raiz do projeto no ``sys.path`` para os scripts acharem ``src``.

Existe para o Colab e para uma maquina limpa: ``python scripts/train.py`` funciona
de qualquer diretorio, sem instalar nada.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
