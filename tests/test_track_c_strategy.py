"""O aluno da trilha C: destilacao com os tres tratamentos de ruido.

Usa um modelo falso de logits controlados e um cache de professor sintetico, para
que o teste exercite a logica do tratamento e nao a capacidade de uma rede.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.config.schema import StrategyConfig
from src.data.loaders import LABELED, UNLABELED
from src.methods import StepOutput, build_strategy
from src.tracks.c.strategy import MODES, DistillationError, soften
from src.tracks.c.teacher import TeacherCache
from tests.contract import assert_cumpre_o_contrato

K = 4
B = 2
MU = 3
IDS = tuple(f"img{i}" for i in range(B * MU))


class LogitsFixos(nn.Module):
    """Devolve logits escolhidos pelo teste, mantendo o grafo de gradiente.

    A estrategia faz **um** forward sobre ``[rotulado, pool]`` concatenados, para que
    a BatchNorm veja os dois ramos no mesmo lote (mesmo motivo da D15 na trilha B).
    O falso reproduz isso e confere o tamanho recebido.
    """

    def __init__(self, *, rotulado: torch.Tensor, pool: torch.Tensor) -> None:
        super().__init__()
        self._logits = torch.cat([rotulado, pool])
        self.escala = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == self._logits.shape[0], (
            f"a estrategia chamou o modelo com lote de {x.shape[0]}; "
            f"o falso foi montado para {self._logits.shape[0]} (rotulado + pool)"
        )
        return self._logits * self.escala


def confiante(classe: int, n: int, forca: float = 20.0) -> torch.Tensor:
    logits = torch.zeros(n, K)
    logits[:, classe] = forca
    return logits


def cache_sintetico(
    probabilidades: np.ndarray, *, teacher: str = "falso"
) -> TeacherCache:
    return TeacherCache(
        teacher=teacher,
        split="train",
        image_ids=IDS[: len(probabilidades)],
        probabilities=probabilidades.astype(np.float32),
        class_names=tuple(f"classe{i}" for i in range(K)),
    )


def professor_certo(confianca: float = 0.9) -> TeacherCache:
    """Professor que sempre diz a classe 0, com a confianca pedida."""
    resto = (1.0 - confianca) / (K - 1)
    linha = np.full(K, resto)
    linha[0] = confianca
    return cache_sintetico(np.tile(linha, (B * MU, 1)))


def montar_batch() -> dict:
    return {
        LABELED: (torch.randn(B, 3), torch.zeros(B, dtype=torch.long)),
        UNLABELED: (torch.randn(B * MU, 3), list(IDS)),
    }


def estrategia(cache: TeacherCache, **kwargs):
    return build_strategy(
        StrategyConfig(name="distillation", params={"cache": cache, **kwargs}),
        num_classes=K,
    )


def modelo(classe: int = 0):
    return LogitsFixos(rotulado=confiante(classe, B), pool=confiante(classe, B * MU))


# ===================== contrato =============================================


@pytest.mark.parametrize("mode", MODES)
def test_cumpre_o_contrato_comum(mode):
    """O mesmo contrato das outras estrategias, verificado aqui porque esta
    precisa do artefato do professor para nascer."""
    assert_cumpre_o_contrato(
        estrategia(professor_certo(), mode=mode), f"distillation/{mode}"
    )


def test_o_pool_e_a_fonte_primaria():
    """As 6.505 imagens do professor definem a epoca; os 105 rotulos sao ciclados.

    Mesmo cronograma da trilha B, para que a comparacao entre trilhas isole a perda.
    """
    assert estrategia(professor_certo()).primary_source == UNLABELED


def test_perda_mantem_o_grafo_para_o_backward():
    saida = estrategia(professor_certo(), mode="soft").training_step(
        modelo(), montar_batch(), device="cpu"
    )
    assert isinstance(saida, StepOutput)
    assert saida.loss.requires_grad
    saida.loss.backward()


# ===================== os tres tratamentos ==================================


def test_hard_usa_todas_as_imagens():
    saida = estrategia(professor_certo(0.4), mode="hard").training_step(
        modelo(), montar_batch(), device="cpu"
    )
    assert saida.components["kept_rate"] == pytest.approx(1.0)


def test_filter_descarta_abaixo_do_limiar():
    """Confianca 0,4 com tau 0,8: nada passa, e a destilacao vira zero."""
    saida = estrategia(professor_certo(0.4), mode="filter", tau=0.8).training_step(
        modelo(), montar_batch(), device="cpu"
    )
    assert saida.components["kept_rate"] == pytest.approx(0.0)
    assert saida.components["distill"] == pytest.approx(0.0)
    assert float(saida.loss.detach()) == pytest.approx(saida.components["ce"])


def test_filter_mantem_acima_do_limiar():
    saida = estrategia(professor_certo(0.9), mode="filter", tau=0.8).training_step(
        modelo(), montar_batch(), device="cpu"
    )
    assert saida.components["kept_rate"] == pytest.approx(1.0)


def test_soft_nunca_descarta_nada():
    """E o ponto do tratamento suave: nao pode colapsar classe, porque nao filtra.

    E a diferenca que a previsao da trilha B (D17) torna testavel.
    """
    saida = estrategia(professor_certo(0.3), mode="soft").training_step(
        modelo(), montar_batch(), device="cpu"
    )
    assert saida.components["kept_rate"] == pytest.approx(1.0)


def test_soft_premia_o_aluno_que_concorda_com_o_professor():
    """KL e zero quando as duas distribuicoes coincidem."""
    s = estrategia(professor_certo(0.9), mode="soft", temperature=1.0)
    concorda = s.training_step(modelo(0), montar_batch(), device="cpu")
    discorda = s.training_step(modelo(1), montar_batch(), device="cpu")
    assert concorda.components["distill"] < discorda.components["distill"]


def test_componentes_reportam_a_confianca_do_professor():
    """O regime de ruido em que o aluno aprendeu fica no history.json."""
    saida = estrategia(professor_certo(0.75), mode="hard").training_step(
        modelo(), montar_batch(), device="cpu"
    )
    assert saida.components["teacher_confidence"] == pytest.approx(0.75, abs=1e-5)


# ===================== temperatura ==========================================


def test_soften_com_t_1_nao_muda_nada():
    p = torch.tensor([[0.7, 0.2, 0.05, 0.05]])
    assert torch.allclose(soften(p, 1.0), p)


def test_soften_aproxima_do_uniforme_e_preserva_a_soma():
    p = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
    suave = soften(p, 4.0)
    assert suave.sum().item() == pytest.approx(1.0)
    assert suave.max().item() < p.max().item()
    assert suave.argmax().item() == p.argmax().item(), "a ordem nao pode mudar"


def test_soften_equivale_a_temperatura_nos_logits():
    """``normalize(p ** (1/T))`` **e** ``softmax(z / T)``, nao uma aproximacao.

    O cache guarda probabilidades, nao logits; sem esta identidade a temperatura
    nao poderia ser aplicada, e o modo `soft` estaria errado.
    """
    z = torch.tensor([[2.0, 0.5, -1.0, 0.25]])
    p = torch.softmax(z, dim=1)
    for t in (0.5, 2.0, 5.0):
        assert torch.allclose(soften(p, t), torch.softmax(z / t, dim=1), atol=1e-6)


# ===================== validacao ============================================


def test_mode_invalido_lista_as_opcoes():
    with pytest.raises(DistillationError, match="mode"):
        estrategia(professor_certo(), mode="inventado")


def test_cache_de_outro_numero_de_classes_e_recusado():
    """Guarda contra apontar o YAML para o .npz de outro conjunto."""
    outro = TeacherCache(
        teacher="falso",
        split="train",
        image_ids=IDS,
        probabilities=np.full((B * MU, K + 1), 1 / (K + 1), dtype=np.float32),
        class_names=tuple(f"c{i}" for i in range(K + 1)),
    )
    with pytest.raises(DistillationError, match="classes"):
        estrategia(outro)


def test_sem_cache_a_mensagem_diz_o_que_fazer():
    """Quem erra isto escreve YAML, nao chama a funcao: TypeError nao serve."""
    with pytest.raises(DistillationError, match="scripts/teacher.py"):
        build_strategy(StrategyConfig(name="distillation"), num_classes=K)


def test_imagem_fora_do_cache_e_recusada_com_instrucao():
    s = estrategia(professor_certo())
    batch = montar_batch()
    batch[UNLABELED] = (batch[UNLABELED][0], ["nao_existe"] * (B * MU))
    with pytest.raises(DistillationError, match="scripts/teacher.py"):
        s.training_step(modelo(), batch, device="cpu")


def test_par_fraco_forte_e_recusado_com_explicacao():
    """`weak_strong` e da trilha B; aqui seria config errado, nao crash obscuro."""
    s = estrategia(professor_certo())
    batch = montar_batch()
    batch[UNLABELED] = ([torch.randn(B * MU, 3)] * 2, list(IDS))
    with pytest.raises(DistillationError, match="weak_strong"):
        s.training_step(modelo(), batch, device="cpu")


def test_cache_incoerente_e_recusado():
    with pytest.raises(Exception, match="incoerente"):
        TeacherCache(
            teacher="falso",
            split="train",
            image_ids=("a", "b"),
            probabilities=np.zeros((3, K), dtype=np.float32),
            class_names=tuple(f"c{i}" for i in range(K)),
        )
