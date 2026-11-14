"""Estrategias de aumentacao registradas."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from src.data.augment import available_augmentations, build_augmentation
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD

TAMANHO = 224


@pytest.fixture
def imagem() -> Image.Image:
    """Imagem com proporcao nao quadrada, como as do pacote apos o corte da faixa."""
    return Image.new("RGB", (512, 341), color=(120, 90, 60))


def test_as_duas_estrategias_da_parte_0_estao_registradas():
    assert set(available_augmentations()) >= {"eval", "basic"}


@pytest.mark.parametrize("nome", ["eval", "basic"])
def test_saida_tem_a_forma_que_o_modelo_espera(nome, imagem):
    tensor = build_augmentation(nome, image_size=TAMANHO)(imagem)
    assert tensor.shape == (3, TAMANHO, TAMANHO)
    assert tensor.dtype == torch.float32


def test_eval_e_deterministica(imagem):
    """A selecao de epoca compara F1 de val entre epocas.

    Se a transformacao de validacao fosse aleatoria, a metrica variaria por motivo
    que nao e o modelo, e a epoca escolhida viraria sorteio.
    """
    transformacao = build_augmentation("eval", image_size=TAMANHO)
    assert torch.equal(transformacao(imagem), transformacao(imagem))


def test_eval_normaliza_com_as_estatisticas_do_imagenet(imagem):
    tensor = build_augmentation("eval", image_size=TAMANHO)(imagem)
    for canal, (media, desvio) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        bruto = {0: 120, 1: 90, 2: 60}[canal] / 255.0
        esperado = (bruto - media) / desvio
        assert tensor[canal].mean().item() == pytest.approx(esperado, abs=1e-3)


def test_imagem_nao_quadrada_nao_e_recortada(imagem):
    """Redimensionar em vez de recortar: o animal pode ocupar 2% do quadro.

    Um CenterCrop descartaria as bordas, e com elas o animal em boa parte das
    imagens. A checagem aqui e que os quatro cantos sobrevivem.
    """
    marcada = imagem.copy()
    for xy in ((0, 0), (511, 0), (0, 340), (511, 340)):
        marcada.putpixel(xy, (255, 255, 255))
    tensor = build_augmentation("eval", image_size=TAMANHO)(marcada)
    fundo = build_augmentation("eval", image_size=TAMANHO)(imagem)
    for canto in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        assert tensor[0][canto] > fundo[0][canto], f"canto {canto} foi recortado"


def test_basic_espelha_mas_preserva_o_conteudo(imagem):
    """Com p=1 o espelhamento acontece; com p=0 a saida iguala a de eval."""
    assorted = Image.new("RGB", (512, 341), color=(0, 0, 0))
    assorted.putpixel((0, 170), (255, 255, 255))

    sem_flip = build_augmentation("basic", image_size=TAMANHO, horizontal_flip=0.0)
    com_flip = build_augmentation("basic", image_size=TAMANHO, horizontal_flip=1.0)
    avaliacao = build_augmentation("eval", image_size=TAMANHO)

    assert torch.equal(sem_flip(assorted), avaliacao(assorted))
    espelhada = com_flip(assorted)
    assert not torch.equal(espelhada, avaliacao(assorted))
    assert torch.equal(espelhada.flip(-1), avaliacao(assorted))


def test_tamanho_vem_do_config(imagem):
    assert build_augmentation("eval", image_size=96)(imagem).shape == (3, 96, 96)
