"""FixMatch — a estrategia da Trilha B.

Usa um modelo falso cuja saida e controlada, para que o teste exercite a logica do
limiar e do alinhamento, e nao a capacidade de uma rede de verdade.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.config.schema import StrategyConfig
from src.data.loaders import LABELED, UNLABELED
from src.methods import StepOutput, TrainingStrategy, build_strategy
from src.tracks.b.strategy import FixMatchStrategy

K = 4
B = 2
MU = 3


class LogitsFixos(nn.Module):
    """Devolve logits que o teste escolhe, mantendo o grafo de gradiente.

    A estrategia faz **um** forward sobre ``[rotulado, fraca, forte]``
    concatenados, para que a BatchNorm veja os tres ramos no mesmo lote. O falso
    reproduz isso: recebe os logits de cada ramo e devolve a concatenacao, na
    mesma ordem.

    O parametro ``escala`` existe so para que ``loss.backward()`` tenha o que
    derivar — sem ele a perda nao estaria ligada a nenhum peso.
    """

    def __init__(
        self, *, rotulado: torch.Tensor, fraca: torch.Tensor, forte: torch.Tensor
    ) -> None:
        super().__init__()
        self._logits = torch.cat([rotulado, fraca, forte])
        self.escala = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == self._logits.shape[0], (
            f"a estrategia chamou o modelo com lote de {x.shape[0]}; "
            f"o falso foi montado para {self._logits.shape[0]} "
            f"(rotulado + fraca + forte)"
        )
        return self._logits * self.escala


def modelo_falso(
    *, rotulado: torch.Tensor | None = None, fraca: torch.Tensor, forte: torch.Tensor
) -> LogitsFixos:
    """Monta o modelo falso a partir dos logits de cada ramo."""
    if rotulado is None:
        rotulado = confiante(0, B)
    return LogitsFixos(rotulado=rotulado, fraca=fraca, forte=forte)


def confiante(classe: int, n: int, forca: float = 20.0) -> torch.Tensor:
    """Logits que concentram quase toda a massa numa classe."""
    logits = torch.zeros(n, K)
    logits[:, classe] = forca
    return logits


def incerto(n: int) -> torch.Tensor:
    """Logits uniformes: confianca 1/K, abaixo de qualquer tau util."""
    return torch.zeros(n, K)


def montar_batch(n_rotulado: int = B, n_sem_rotulo: int = B * MU) -> dict:
    return {
        LABELED: (
            torch.randn(n_rotulado, 3),
            torch.zeros(n_rotulado, dtype=torch.long),
        ),
        UNLABELED: (
            [torch.randn(n_sem_rotulo, 3), torch.randn(n_sem_rotulo, 3)],
            [f"img{i}" for i in range(n_sem_rotulo)],
        ),
    }


def estrategia(**kwargs) -> FixMatchStrategy:
    return build_strategy(StrategyConfig(name="fixmatch", params=kwargs), num_classes=K)


def test_esta_registrada_e_cumpre_o_contrato():
    s = estrategia()
    assert isinstance(s, TrainingStrategy)
    assert s.required_sources == (LABELED, UNLABELED)


def test_pool_sem_rotulo_e_a_fonte_primaria():
    """A epoca varre as 6.505 imagens e cicla os 105 rotulos.

    Se fosse o contrario, a epoca acabaria em 7 lotes e o pool quase nao seria
    visto — o FixMatch perderia o que o torna FixMatch.
    """
    assert estrategia().primary_source == UNLABELED


def test_tudo_confiante_passa_o_limiar():
    modelo = modelo_falso(fraca=confiante(0, B * MU), forte=confiante(0, B * MU))
    saida = estrategia(tau=0.8).training_step(modelo, montar_batch(), device="cpu")
    assert isinstance(saida, StepOutput)
    assert saida.components["mask_rate"] == pytest.approx(1.0)


def test_nada_confiante_anula_a_consistencia():
    """Com tudo abaixo de tau, a perda vira so a supervisionada.

    E o modo de falha que mais preocupa aqui: com 21 classes e 105 rotulos o
    modelo e pouco confiante, e um tau alto demais faz o FixMatch nao treinar
    nada alem do supervisionado — sem erro nenhum aparecer no log.
    """
    modelo = modelo_falso(fraca=incerto(B * MU), forte=incerto(B * MU))
    s = estrategia(tau=0.8)
    saida = s.training_step(modelo, montar_batch(), device="cpu")

    assert saida.components["mask_rate"] == pytest.approx(0.0)
    assert saida.components["consistency"] == pytest.approx(0.0)
    assert float(saida.loss.detach()) == pytest.approx(saida.components["ce"])


def test_lambda_zero_equivale_a_supervisionado():
    modelo = modelo_falso(fraca=confiante(0, B * MU), forte=confiante(0, B * MU))
    saida = estrategia(tau=0.1, lambda_u=0.0).training_step(
        modelo, montar_batch(), device="cpu"
    )
    assert float(saida.loss.detach()) == pytest.approx(saida.components["ce"])


def test_perda_mantem_o_grafo_para_o_backward():
    modelo = modelo_falso(fraca=confiante(0, B * MU), forte=confiante(0, B * MU))
    saida = estrategia(tau=0.5).training_step(modelo, montar_batch(), device="cpu")
    assert saida.loss.requires_grad
    saida.loss.backward()
    assert modelo.escala.grad is not None


def test_denominador_e_mu_b_e_nao_o_numero_que_passou():
    """O paper divide a perda nao supervisionada por mu*B, fixo.

    Com denominador variavel, o peso do termo cresceria sozinho conforme o modelo
    ganha confianca, e o ajuste de lambda_u deixaria de significar o mesmo ao
    longo do treino.
    """
    # metade das vistas fracas confiante, metade nao: 3 de 6 passam
    metade = torch.cat([confiante(1, B * MU // 2), incerto(B * MU - B * MU // 2)])
    modelo = modelo_falso(fraca=metade, forte=confiante(1, B * MU))
    saida = estrategia(tau=0.8).training_step(modelo, montar_batch(), device="cpu")
    assert saida.components["mask_rate"] == pytest.approx(0.5)


# ===================== alinhamento de distribuicao ==========================


def test_sem_alinhamento_a_media_movel_fica_parada():
    s = estrategia(distribution_alignment=False)
    antes = s.state_dict()["running"].clone()
    modelo = modelo_falso(fraca=confiante(0, B * MU), forte=confiante(0, B * MU))
    s.training_step(modelo, montar_batch(), device="cpu")
    assert torch.allclose(s.state_dict()["running"], antes)


def test_alinhamento_move_a_media_para_a_classe_predita():
    s = estrategia(distribution_alignment=True, da_momentum=0.0, tau=0.5)
    modelo = modelo_falso(fraca=confiante(2, B * MU), forte=confiante(2, B * MU))
    s.training_step(modelo, montar_batch(), device="cpu")
    media = s.state_dict()["running"]
    assert media.argmax().item() == 2


def test_alinhamento_desloca_o_pseudo_rotulo_para_a_classe_sub_representada():
    """O ponto do DA: se o modelo so chuta a classe 0, realinhar puxa para as outras.

    Monta uma media movel saturada na classe 0 e mostra que uma predicao
    ligeiramente favoravel a 0 passa a virar pseudo-rotulo de outra classe.
    """
    s = estrategia(distribution_alignment=True)
    saturada = torch.tensor([0.97, 0.01, 0.01, 0.01])
    s.load_state_dict({"running": saturada})

    probabilidades = torch.tensor([[0.40, 0.30, 0.15, 0.15]])
    alinhada = s._align(probabilidades)

    assert probabilidades.argmax().item() == 0
    assert alinhada.argmax().item() == 1
    assert alinhada.sum().item() == pytest.approx(1.0)


def test_media_movel_sobrevive_ao_resume():
    """Vai para `strategy_state` no checkpoint; sem ela o DA reiniciaria em uniforme."""
    s = estrategia(distribution_alignment=True)
    s.load_state_dict({"running": torch.tensor([0.7, 0.1, 0.1, 0.1])})
    outra = estrategia(distribution_alignment=True)
    outra.load_state_dict(s.state_dict())
    assert torch.allclose(outra.state_dict()["running"], s.state_dict()["running"])


def test_estado_ausente_nao_quebra_o_resume():
    s = estrategia(distribution_alignment=True)
    s.load_state_dict({})
    assert s.state_dict()["running"].sum().item() == pytest.approx(1.0)


# ===================== validacao de parametros ==============================


@pytest.mark.parametrize("tau", [0.0, -0.1, 1.5])
def test_tau_fora_da_faixa_e_recusado(tau):
    with pytest.raises(ValueError, match="tau"):
        estrategia(tau=tau)


def test_lambda_negativo_e_recusado():
    with pytest.raises(ValueError, match="lambda_u"):
        estrategia(lambda_u=-1.0)


def test_momento_fora_da_faixa_e_recusado():
    with pytest.raises(ValueError, match="da_momentum"):
        estrategia(da_momentum=1.0)
