"""O proxy de "foto noturna": saturacao media da propria imagem.

Mora na infraestrutura neutra, e nao numa trilha, porque a trilha C o usa para dizer
onde o professor erra e a Parte 3 o usa para diagnosticar os erros do modelo entregue.
Apagar uma trilha nao pode levar a medida embora.
"""

from __future__ import annotations

from PIL import Image

from src.analysis.images import NIGHT_SATURATION_THRESHOLD, is_night, mean_saturation


def test_imagem_em_tons_de_cinza_tem_saturacao_zero():
    """E o caso real: camera trap noturna grava em infravermelho, sem cor."""
    cinza = Image.new("L", (8, 8), color=90).convert("RGB")
    assert mean_saturation(cinza) == 0.0
    assert is_night(cinza)


def test_imagem_de_cor_saturada_esta_longe_do_limiar():
    vermelho = Image.new("RGB", (8, 8), color=(255, 0, 0))
    assert mean_saturation(vermelho) == 255.0
    assert not is_night(vermelho)


def test_o_limiar_e_baixo_porque_infravermelho_nao_tem_cor():
    assert 0 < NIGHT_SATURATION_THRESHOLD < 128
