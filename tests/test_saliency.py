"""CAM: a decomposicao, a igualdade com Grad-CAM, as medidas e a oclusao.

O teste que importa mais e ``test_cam_e_igual_a_gradcam_a_menos_de_uma_constante``. O
enunciado pede "Grad-CAM ou oclusao" e nos usamos CAM, que e o que esta nas aulas; a
afirmacao de que os dois coincidem numa rede com GAP + linear nao e citada de lugar
nenhum, e verificada aqui contra o autograd.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.analysis.saliency import (
    SaliencyError,
    check_decomposition,
    class_activation_maps,
    coldest_mask,
    decompose,
    faithfulness,
    heatmaps,
    map_statistics,
    mask_iou,
    scan,
    top_mass_mask,
)


class RedeComGap(nn.Module):
    """Convolucao, global average pooling e uma linear — a forma que o CAM exige.

    Escrita a mao e nao com ``nn.Sequential`` porque e assim que a ResNet-18 e: o
    achatamento acontece dentro do ``forward``, e nao como submodulo.
    """

    def __init__(self, canais: int = 4, classes: int = 3) -> None:
        super().__init__()
        self.corpo = nn.Conv2d(3, canais, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(canais, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(self.corpo(x))
        return self.fc(torch.flatten(x, 1))


class RedeQueOlhaUmCanto(nn.Module):
    """Logit da classe 0 = media dos pixels no quadrante superior esquerdo.

    Serve para testar a **oclusao**: aqui sabemos, por construcao, onde a decisao e
    tomada, entao da para checar se o teste de fidelidade descobre isso sozinho.

    Media e nao soma de proposito. Com a soma, o logit passa de 300 e a softmax
    satura em 1,0 — qualquer oclusao deixaria a probabilidade intacta e o teste
    mediria a saturacao da softmax, nao a fidelidade do mapa.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        altura, largura = x.shape[-2:]
        canto = x[:, :, : altura // 2, : largura // 2].mean(dim=(1, 2, 3))
        return torch.stack([canto, torch.zeros_like(canto)], dim=1)


def mapa_de_canto(lado: int, quente: slice) -> np.ndarray:
    mapa = np.zeros((lado, lado))
    mapa[quente, quente] = 1.0
    return mapa


# ===================== a decomposicao =======================================


def test_a_decomposicao_reproduz_a_resnet18_de_verdade():
    """O teste nao vale num modelo de brinquedo: e a ResNet-18 que vai ser explicada.

    Sem pesos de ImageNet (``pretrained=False``), que seriam um download; a igualdade
    testada e entre dois caminhos de forward do **mesmo** modelo, e nao depende de
    quais pesos ele tem.
    """
    from src.config.schema import ModelConfig
    from src.models import build_model

    modelo = build_model(
        ModelConfig(name="resnet18", pretrained=False), num_classes=21
    ).eval()
    diferenca = check_decomposition(
        modelo, decompose(modelo), torch.randn(2, 3, 224, 224)
    )
    assert diferenca < 1e-4


def test_decompose_recusa_modelo_que_nao_termina_em_linear():
    modelo = nn.Sequential(nn.Conv2d(3, 4, 1), nn.AdaptiveAvgPool2d(1), nn.ReLU())
    with pytest.raises(SaliencyError):
        decompose(modelo)


def test_decompose_recusa_modelo_sem_average_pooling():
    """Sem GAP os pesos da linear nao sao os w_k^c: falhar alto e melhor que um mapa."""
    modelo = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Conv2d(4, 4, 1), nn.Linear(4, 3))
    with pytest.raises(SaliencyError):
        decompose(modelo)


def test_decompose_aceita_flatten_entre_o_pooling_e_a_linear():
    modelo = nn.Sequential(
        nn.Conv2d(3, 4, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3)
    )
    cam = decompose(modelo)
    assert cam.num_classes == 3
    assert check_decomposition(modelo, cam, torch.randn(2, 3, 8, 8)) < 1e-5


# ===================== CAM == Grad-CAM ======================================


def test_cam_e_igual_a_gradcam_a_menos_de_uma_constante():
    """alpha_k^c = w_k^c / Z, logo os dois mapas sao proporcionais com fator 1/Z.

    O Grad-CAM e calculado aqui pelo caminho dele mesmo — autograd sobre o logit —,
    sem reusar nada do modulo que esta sendo testado.
    """
    torch.manual_seed(0)
    modelo = RedeComGap(canais=5, classes=4).eval()
    cam = decompose(modelo)
    imagem = torch.randn(1, 3, 7, 7)
    classe = 2

    ativacoes = cam.forward_features(imagem).detach().requires_grad_(True)
    logit = cam.logits_from_features(ativacoes)[0, classe]
    (gradiente,) = torch.autograd.grad(logit, ativacoes)
    alfa = gradiente.mean(dim=(2, 3))  # (1, k)
    mapa_gradcam = torch.relu(torch.einsum("nk,nkhw->nhw", alfa, ativacoes.detach()))

    mapa_cam = torch.relu(
        class_activation_maps(ativacoes.detach(), cam.weight, [classe])
    )
    posicoes = ativacoes.shape[2] * ativacoes.shape[3]
    assert torch.allclose(mapa_gradcam * posicoes, mapa_cam, atol=1e-5)
    # e o que de fato usamos, o mapa normalizado, e identico mesmo sem a constante
    assert torch.allclose(
        heatmaps(mapa_gradcam, size=7), heatmaps(mapa_cam, size=7), atol=1e-5
    )


# ===================== o mapa ===============================================


def test_heatmaps_amplia_e_normaliza_pelo_pico():
    bruto = torch.tensor([[[-5.0, 1.0], [2.0, 4.0]]])
    calor = heatmaps(bruto, size=8)
    assert calor.shape == (1, 8, 8)
    assert float(calor.max()) == pytest.approx(1.0)
    assert float(calor.min()) >= 0.0, "a ReLU descarta a evidencia contraria"


def test_mapa_sem_valor_positivo_vira_zeros_sem_dividir_por_zero():
    calor = heatmaps(torch.full((1, 3, 3), -2.0), size=6)
    assert float(calor.abs().max()) == 0.0


def test_class_activation_maps_recusa_uma_classe_por_imagem_errada():
    with pytest.raises(SaliencyError):
        class_activation_maps(torch.zeros(2, 4, 3, 3), torch.zeros(5, 4), [0])


# ===================== as medidas ===========================================


def test_concentracao_de_mapa_uniforme_e_a_propria_fracao():
    """Um mapa sem estrutura concentra nos 10% mais quentes exatamente 10% da massa.

    E o valor nulo contra o qual a etiqueta "atencao difusa" se compara.
    """
    medidas = map_statistics(np.ones((20, 20)), top_fraction=0.10)
    assert medidas.concentration == pytest.approx(0.10, abs=1e-9)


def test_mapa_num_canto_tem_massa_na_borda_acima_da_area_dela():
    medidas = map_statistics(mapa_de_canto(20, slice(0, 4)))
    assert medidas.border_mass == pytest.approx(1.0)
    assert medidas.border_excess > 0


def test_mapa_no_centro_tem_massa_na_borda_abaixo_da_area_dela():
    mapa = np.zeros((20, 20))
    mapa[8:12, 8:12] = 1.0
    medidas = map_statistics(mapa)
    assert medidas.border_mass == pytest.approx(0.0)
    assert medidas.border_excess < 0


def test_pixels_quentes_viram_o_lado_de_um_quadrado_equivalente():
    mapa = np.zeros((20, 20))
    mapa[0:4, 0:4] = 1.0
    medidas = map_statistics(mapa)
    assert medidas.hot_pixels == 16
    assert medidas.hot_side == pytest.approx(4.0)


def test_top_mass_mask_tem_sempre_a_area_pedida():
    """Area fixa e o que torna comparavel ocluir a regiao quente e a fria."""
    aleatorio = np.random.default_rng(0).random((30, 30))
    for fracao in (0.05, 0.10, 0.25):
        assert top_mass_mask(aleatorio, fraction=fracao).sum() == round(
            fracao * aleatorio.size
        )


def test_mapa_constante_nao_faz_a_mascara_pegar_tudo():
    """Com empate em todos os pixels, a mascara ainda tem de respeitar a area."""
    assert top_mass_mask(np.ones((10, 10)), fraction=0.10).sum() == 10


def test_a_mascara_fria_fica_longe_do_pico_quando_ha_empate():
    """O desempate pela distancia e o que impede o controle de virar a faixa de cima.

    Com o mapa quente num canto e zero no resto, "os 10% mais frios" empata em quase
    todos os pixels. Na ordem do array isso daria as primeiras linhas da imagem — do
    mesmo lado do pico. O controle tem de cair do lado oposto.
    """
    mapa = mapa_de_canto(20, slice(0, 4))
    fria = coldest_mask(mapa, fraction=0.10)
    linhas, colunas = np.nonzero(fria)
    assert fria.sum() == 40
    assert linhas.mean() > 10 and colunas.mean() > 10, "o controle ficou junto do pico"
    assert not (fria & top_mass_mask(mapa, fraction=0.10)).any()


def test_mask_iou():
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[:2, :2] = True
    b[1:3, 1:3] = True
    assert mask_iou(a, a) == pytest.approx(1.0)
    assert mask_iou(a, b) == pytest.approx(1 / 7)
    assert mask_iou(a, np.zeros((4, 4), dtype=bool)) == 0.0


# ===================== o mapa explica a decisao? ============================


def test_oclusao_confirma_um_mapa_fiel():
    """O modelo decide pelo canto superior esquerdo; o mapa aponta para la."""
    modelo = RedeQueOlhaUmCanto().eval()
    imagens = torch.ones(1, 3, 20, 20)
    mapa = mapa_de_canto(20, slice(0, 6))
    (sonda,) = faithfulness(modelo, imagens, [mapa], [0], device="cpu", fraction=0.10)
    assert sonda.drop_hot() > 0, "apagar a regiao apontada tem de custar confianca"
    assert sonda.explains()


def test_oclusao_desmente_um_mapa_que_aponta_para_o_lugar_errado():
    """Mesmo modelo, mapa mentiroso: o controle cai mais que a regiao "quente"."""
    modelo = RedeQueOlhaUmCanto().eval()
    imagens = torch.ones(1, 3, 20, 20)
    mapa_errado = np.zeros((20, 20))
    mapa_errado[14:, 14:] = 1.0  # aponta para o canto oposto ao que decide
    (sonda,) = faithfulness(
        modelo, imagens, [mapa_errado], [0], device="cpu", fraction=0.10
    )
    assert sonda.drop_hot() == pytest.approx(0.0, abs=1e-6)
    assert not sonda.explains()


def test_faithfulness_recusa_quantidades_incompativeis():
    with pytest.raises(SaliencyError):
        faithfulness(
            RedeQueOlhaUmCanto(),
            torch.ones(2, 3, 8, 8),
            [np.ones((8, 8))],
            [0, 0],
            device="cpu",
        )


# ===================== a varredura ==========================================


def test_scan_devolve_probabilidades_iguais_ao_forward_do_modelo():
    torch.manual_seed(1)
    modelo = RedeComGap(canais=4, classes=3).eval()
    cam = decompose(modelo)
    imagens = torch.randn(5, 3, 8, 8)
    dados = [(imagens[i], i % 3) for i in range(5)]
    varredura = scan(cam, DataLoader(dados, batch_size=2), device="cpu", size=8)

    esperado = torch.softmax(modelo(imagens), dim=1).detach().numpy()
    assert np.allclose(varredura.probabilities, esperado, atol=1e-5)
    assert varredura.y_true.tolist() == [0, 1, 2, 0, 1]
    assert len(varredura.statistics) == 5
    assert varredura.measure("concentration").shape == (5,)
    assert varredura.correct().shape == (5,)
