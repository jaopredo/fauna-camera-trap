"""Medidas tiradas da propria imagem, sem rotulo e sem modelo.

Existe por uma razao de arquitetura, nao de conveniencia. O proxy de "foto noturna"
nasceu na trilha C (o obrigatorio 2 pede dizer se o professor erra mais a noite), mas
a **Parte 3** precisa do mesmo proxy para diagnosticar os erros do modelo entregue — e
a Parte 3 nao pertence a nenhuma trilha. Deixa-lo em ``src/tracks/c/`` faria apagar a
trilha C quebrar a Parte 3, contra a propriedade que ``docs/decisions.md`` D18
promete. Mesmo motivo pelo qual a aumentacao ``weak`` vive em ``src/data`` e nao na
trilha B: quem usa esta fora da trilha.

Regra de escopo deste modulo: **so o que se mede da imagem crua**, sem rotulo humano,
sem modelo e sem metadado externo. Qualquer medida que precise de rotulo e de
``src.metrics``; qualquer medida que precise do modelo e de ``src.analysis.saliency``.
"""

from __future__ import annotations

import numpy as np

#: Abaixo desta saturacao media (escala 0-255) a imagem e tratada como infravermelha.
#: Camera trap noturna ilumina com IR e grava praticamente sem cor, entao a saturacao
#: media fica perto de zero; fotos diurnas desbotadas ainda ficam bem acima.
NIGHT_SATURATION_THRESHOLD = 25.0


def mean_saturation(image) -> float:
    """Saturacao media dos pixels (0-255). Proxy de "foto noturna em IR".

    Recebe uma imagem PIL para nao obrigar este modulo a conhecer caminho de arquivo
    nem ``DataLoader``. O ``convert`` e necessario porque parte do pacote vem em tons
    de cinza, com um canal so, e o canal S do HSV nao existe nesse modo.
    """
    return float(np.asarray(image.convert("HSV"))[:, :, 1].mean())


def is_night(image) -> bool:
    """Verdadeiro quando a imagem parece ter sido tirada com iluminacao IR."""
    return mean_saturation(image) < NIGHT_SATURATION_THRESHOLD
