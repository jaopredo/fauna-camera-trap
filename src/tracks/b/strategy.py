"""FixMatch — a estrategia da Trilha B.

Segue a aula 13 (``DL_20242_13_Unsupervised.pdf``, p. 85-91), que o define como
"a simple combination of two common approaches to SSL: consistency regularization
and pseudo-labeling" (p. 85):

* p. 88 — entropia cruzada supervisionada sobre a vista **fraca** dos rotulados;
* p. 89 — para cada imagem sem rotulo, guarda a versao **fortemente** aumentada e
  calcula o pseudo-rotulo a partir da **fracamente** aumentada;
* p. 90 — perda nao supervisionada com **pseudo-rotulo duro** e **limiar de
  ativacao** tau, como regularizacao de consistencia;
* p. 91 — perda final como soma ponderada das duas, pelo peso lambda_u (p. 87).

O pool sem rotulo e a fonte primaria: uma epoca varre as 6.505 imagens e cicla os
105 rotulos. Se fosse o contrario, a epoca acabaria em 7 lotes e o pool quase nao
seria visto.

**Sobre tau.** O paper usa 0,95, calibrado em CIFAR-10: 10 classes balanceadas e um
modelo confiante. Medimos aqui, no ``val``, que com 21 classes e 105 rotulos esse
limiar deixa **6 das 21 classes sem nenhum pseudo-rotulo**. Por isso o padrao deste
modulo e mais baixo, e a escolha final sai do ``val`` (ver ``docs/decisions.md``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from src.data.loaders import LABELED, UNLABELED
from src.methods.base import Batch, StepOutput, TrainingStrategy
from src.methods.registry import register_strategy

if TYPE_CHECKING:
    from torch import nn as _nn

#: Limiar padrao. Abaixo do 0,95 do paper, pelo motivo no docstring do modulo.
DEFAULT_TAU = 0.80

#: Peso da perda nao supervisionada. 1,0 e o valor do paper.
DEFAULT_LAMBDA_U = 1.0

#: Momento da media movel do alinhamento de distribuicao.
DEFAULT_DA_MOMENTUM = 0.999


class FixMatchStrategy(TrainingStrategy):
    """Consistencia entre vista fraca e forte, com pseudo-rotulo acima de tau."""

    required_sources = (LABELED, UNLABELED)
    #: o pool sem rotulo define o tamanho da epoca; os 105 rotulos sao ciclados
    primary_source = UNLABELED

    def __init__(
        self,
        *,
        num_classes: int,
        tau: float = DEFAULT_TAU,
        lambda_u: float = DEFAULT_LAMBDA_U,
        distribution_alignment: bool = False,
        da_momentum: float = DEFAULT_DA_MOMENTUM,
    ) -> None:
        if not 0.0 < tau <= 1.0:
            raise ValueError(f"tau precisa estar em (0, 1], veio {tau}")
        if lambda_u < 0:
            raise ValueError(f"lambda_u nao pode ser negativo, veio {lambda_u}")
        if not 0.0 <= da_momentum < 1.0:
            raise ValueError(f"da_momentum precisa estar em [0, 1), veio {da_momentum}")

        self.num_classes = num_classes
        self.tau = tau
        self.lambda_u = lambda_u
        self.distribution_alignment = distribution_alignment
        self.da_momentum = da_momentum
        self._supervised_loss = nn.CrossEntropyLoss()
        # media movel das predicoes na vista fraca; comeca uniforme para que o
        # realinhamento seja neutro antes de haver estatistica acumulada
        self._running = torch.full((num_classes,), 1.0 / num_classes)

    # ------------------------------------------------------------------ estado
    def state_dict(self) -> dict[str, Any]:
        """A media movel do DA precisa voltar no ``--resume``.

        Sem ela, retomar um treino reiniciaria o realinhamento em uniforme e o
        modelo levaria centenas de passos para reconstruir a estatistica.
        """
        return {"running": self._running.clone()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        guardada = state.get("running")
        if guardada is not None:
            self._running = torch.as_tensor(guardada).float().cpu()

    # ------------------------------------------------------------ alinhamento
    def _align(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Realinha as probabilidades para a distribuicao alvo.

        ``q' = normalize(q * alvo / media_movel)``. O alvo e **uniforme**: o
        orcamento de rotulos e uniforme por construcao (5 por classe), nao
        conhecemos o prior do pool sem rotulo sem rotula-lo, e o F1 macro pesa
        cavalo igual a urso — empurrar para o balanceado e alinhado a metrica.

        Tecnica fora do material da disciplina: o enunciado a oferece pelo nome,
        mas ela nao aparece nas aulas 11-15.
        """
        media = self._running.to(probabilities.device).clamp_min(1e-6)
        alvo = 1.0 / self.num_classes
        realinhada = probabilities * (alvo / media)
        return realinhada / realinhada.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def _update_running(self, probabilities: torch.Tensor) -> None:
        lote = probabilities.mean(dim=0).detach().cpu()
        self._running = self.da_momentum * self._running + (1 - self.da_momentum) * lote

    # -------------------------------------------------------------------- passo
    def training_step(
        self, model: "_nn.Module", batch: Batch, *, device: Any
    ) -> StepOutput:
        images, labels = batch[LABELED]
        views, _ids = batch[UNLABELED]
        weak, strong = views[0].to(device), views[1].to(device)
        images, labels = images.to(device), labels.to(device)

        # UM forward sobre os tres ramos concatenados. Nao e otimizacao, e
        # correcao: com forwards separados a BatchNorm atualiza as estatisticas
        # acumuladas duas vezes por passo, com distribuicoes diferentes, e o
        # pseudo-rotulo sai calculado sob estatisticas de lote que nao sao as
        # mesmas da vista forte contra a qual ele e aplicado — alvo e predicao
        # passariam a viver em normalizacoes distintas.
        n_rot, n_sem = images.size(0), weak.size(0)
        logits = model(torch.cat([images, weak, strong]))
        logits_labeled = logits[:n_rot]
        logits_weak = logits[n_rot : n_rot + n_sem]
        logits_strong = logits[n_rot + n_sem :]

        supervised = self._supervised_loss(logits_labeled, labels)

        # o pseudo-rotulo e alvo, nao predicao (aula p. 89): detach corta o
        # gradiente sem exigir um forward separado
        with torch.no_grad():
            probabilities = torch.softmax(logits_weak.detach().float(), dim=1)
            if self.distribution_alignment:
                self._update_running(probabilities)
                probabilities = self._align(probabilities)
            confidence, pseudo = probabilities.max(dim=1)
            mask = (confidence >= self.tau).float()

        por_imagem = F.cross_entropy(logits_strong, pseudo, reduction="none")
        # divide por mu*B, nao pelo numero de imagens que passaram: e assim que o
        # paper define a perda, e manter o denominador fixo faz o peso do termo
        # crescer conforme o modelo ganha confianca, em vez de ficar constante
        consistency = (por_imagem * mask).mean()

        loss = supervised + self.lambda_u * consistency
        return StepOutput(
            loss=loss,
            components={
                "ce": float(supervised.detach()),
                "consistency": float(consistency.detach()),
                # fracao do lote que passou tau — o sinal que diz se o FixMatch
                # tem materia-prima; perto de zero significa treino sem efeito
                "mask_rate": float(mask.mean().detach()),
            },
        )


@register_strategy("fixmatch")
def build_fixmatch(
    *,
    num_classes: int,
    tau: float = DEFAULT_TAU,
    lambda_u: float = DEFAULT_LAMBDA_U,
    distribution_alignment: bool = False,
    da_momentum: float = DEFAULT_DA_MOMENTUM,
    **_: object,
) -> TrainingStrategy:
    """Constroi o FixMatch com os parametros declarados no YAML."""
    return FixMatchStrategy(
        num_classes=num_classes,
        tau=tau,
        lambda_u=lambda_u,
        distribution_alignment=distribution_alignment,
        da_momentum=da_momentum,
    )
