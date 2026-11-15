"""As trilhas do enunciado, uma subpasta cada, descobertas automaticamente.

O enunciado oferece tres trilhas (A: auto-supervisionado, B: semi-supervisionado,
C: professor e aluno) e permite escolher uma. Implementamos B e C para compara-las.

**Por que descoberta automatica, e so aqui.** Em todo o resto do projeto os imports
sao explicitos. Nesta pasta, nao: a regra e que **apagar uma trilha seja apagar a
pasta dela**, sem editar nenhum arquivo compartilhado. Com um import explicito
(``from src.tracks import b, c``) remover a trilha C quebraria o import da B, e as
duas deixariam de ser independentes de fato.

**O que uma trilha pode conter.** A estrategia de treino, os diagnosticos dela, as
figuras dela, e nada mais. A infraestrutura neutra — configuracao, dados, modelos,
laco de treino, avaliacao, metricas, ``src/analysis/budget.py`` — nao pertence a
trilha nenhuma e fica fora.

**O que uma trilha nao pode fazer.** Importar de outra trilha.
``tests/test_tracks_independence.py`` percorre o grafo de imports e falha se isso
acontecer, para que a independencia seja uma assercao e nao uma promessa.
"""

from __future__ import annotations

import importlib
import pkgutil

#: Nomes das trilhas presentes, em ordem. Vazio se nenhuma estiver instalada, o que
#: e um estado valido: o projeto roda a Parte 0 sem trilha alguma.
AVAILABLE_TRACKS: tuple[str, ...] = tuple(
    sorted(info.name for info in pkgutil.iter_modules(__path__) if info.ispkg)
)

for _nome in AVAILABLE_TRACKS:
    # importar a subpasta roda os decoradores (@register_strategy,
    # @register_augmentation) que registram o que a trilha oferece
    importlib.import_module(f"{__name__}.{_nome}")

__all__ = ["AVAILABLE_TRACKS"]
