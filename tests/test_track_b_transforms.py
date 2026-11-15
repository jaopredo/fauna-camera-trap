"""O par fraco/forte do FixMatch — aumentacao da trilha B.

A vista fraca sozinha e testada em ``tests/test_transforms.py``: ela e
infraestrutura compartilhada, porque o modelo entregue a usa.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from src.data.augment import available_augmentations, build_augmentation

TAMANHO = 224


@pytest.fixture
def imagem() -> Image.Image:
    """Imagem com proporcao nao quadrada, como as do pacote apos o corte da faixa."""
    return Image.new("RGB", (512, 341), color=(120, 90, 60))


def test_weak_strong_esta_registrada():
    assert "weak_strong" in available_augmentations()


def test_weak_strong_devolve_duas_vistas(imagem):
    """A aula 13 (p. 89) pede o par vindo da MESMA imagem."""
    fraca, forte = build_augmentation("weak_strong", image_size=TAMANHO)(imagem)
    assert fraca.shape == forte.shape == (3, TAMANHO, TAMANHO)
    assert not torch.equal(fraca, forte), "as duas vistas nao podem ser identicas"


def test_weak_perturba_menos_que_strong(imagem):
    """A vista fraca gera o pseudo-rotulo; se ela destruisse a imagem como a
    forte, o alvo seria ruido e o FixMatch nao teria o que propagar."""
    import numpy as np

    referencia = build_augmentation("eval", image_size=TAMANHO)(imagem)
    par = build_augmentation(
        "weak_strong", image_size=TAMANHO, horizontal_flip=0.0, cutout=1.0
    )
    distancias_fracas, distancias_fortes = [], []
    for _ in range(12):
        fraca, forte = par(imagem)
        distancias_fracas.append(float((fraca - referencia).abs().mean()))
        distancias_fortes.append(float((forte - referencia).abs().mean()))
    assert np.mean(distancias_fortes) > np.mean(distancias_fracas)


def test_weak_strong_e_serializavel():
    """Com num_workers > 0 o DataLoader envia a transformacao para os filhos,
    e closure aninhada nao sobrevive ao pickle."""
    import pickle

    par = build_augmentation("weak_strong", image_size=TAMANHO)
    assert pickle.loads(pickle.dumps(par)) is not None
