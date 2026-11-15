"""O aluno da trilha C: destilacao dos pseudo-rotulos do professor.

O enunciado: "Um foundation model aberto rotula o train e um modelo pequeno (o
aluno) e treinado nesses pseudo-rotulos, junto com os 105 rotulos verdadeiros."

    perda = CE(105 rotulos verdadeiros) + lambda_u * destilacao(6.505 do professor)

**Os tres tratamentos de ruido.** O obrigatorio 3 pede tratar o ruido do professor e
oferece opcoes. Implementamos tres, selecionadas por ``mode``, porque a comparacao
entre elas e o experimento:

``hard``    pseudo-rotulo duro, sem tratamento nenhum. E a referencia: mostra quanto
            ruido o professor injeta quando ninguem faz nada.
``filter``  descarta as imagens com confianca abaixo de ``tau``. E a escolha
            ingenua — e a trilha B prediz que ela vai colapsar classes.
``soft``    rotulos suaves com temperatura, sobre **todas** as imagens. Nao descarta
            nada, logo nao pode filtrar classe nenhuma.

**A previsao que estamos testando.** Na trilha B medimos que um limiar de confianca
nao filtra *erro*, filtra *incerteza* — e a incerteza se concentra nas classes que o
modelo confunde. O limiar virou um filtro de classe disfarcado, e o F1 macro, que
pesa cavalo igual a urso, puniu isso (ver ``docs/decisions.md``, D17). Se a previsao
valer tambem aqui, ``filter`` perde de ``soft`` em F1 macro apesar de treinar com
rotulos mais limpos. E uma previsao falsificavel feita antes de rodar.

**Sobre a taxa de aprendizado.** Esta estrategia usa o pool como fonte primaria,
igual ao FixMatch: ~58 passos por epoca em vez de 7, logo ~9 passagens pelos 105
rotulos por epoca. A D16 mostrou que a mesma lr da Parte 0 memoriza 4x mais nesse
regime. Os configs da trilha C ja nascem com ``lr: 0.00003``, nao com 1e-4 —
aprendemos isso caro na trilha B.

**Fora do material da disciplina.** Destilacao professor-aluno nao aparece nas aulas
11-15. O enunciado a descreve e autoriza; as aulas nao a ensinam.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.data.loaders import LABELED, UNLABELED
from src.errors import AppError
from src.methods.base import Batch, StepOutput, TrainingStrategy
from src.methods.registry import register_strategy
from src.tracks.c.teacher import TeacherCache

if TYPE_CHECKING:
    from torch import nn as _nn

#: Tratamentos de ruido disponiveis, na ordem em que a apresentacao os compara.
MODES = ("hard", "filter", "soft")

#: Limiar do modo ``filter``. 0,80 e o mesmo da trilha B, de proposito: a previsao
#: que estamos testando veio de la, e mudar o limiar junto confundiria a comparacao.
DEFAULT_TAU = 0.80

#: Temperatura do modo ``soft``. Acima de 1 suaviza a distribuicao do professor,
#: espalhando massa para as classes que ele considerou em segundo lugar — que e
#: exatamente o sinal que o pseudo-rotulo duro joga fora.
DEFAULT_TEMPERATURE = 2.0

#: Peso do termo de destilacao.
DEFAULT_LAMBDA_U = 1.0


class DistillationError(AppError, ValueError):
    """Configuracao incoerente da destilacao."""


def soften(probabilities: torch.Tensor, temperature: float) -> torch.Tensor:
    r"""Aplica temperatura a uma distribuicao **ja normalizada**.

    O cache guarda probabilidades, nao logits, e isso parece impedir a temperatura.
    Nao impede: como :math:`p = \mathrm{softmax}(z)`, vale

    .. math:: p^{1/T} \propto \exp(z)^{1/T} = \exp(z/T) \propto \mathrm{softmax}(z/T)

    e o softmax e invariante a deslocamento. Logo ``normalize(p ** (1/T))`` **e**
    exatamente ``softmax(z / T)``. Nao e aproximacao.
    """
    if temperature <= 0:
        raise DistillationError(f"temperature precisa ser > 0, veio {temperature}")
    if temperature == 1.0:
        return probabilities
    suavizada = probabilities.clamp_min(1e-12).pow(1.0 / temperature)
    return suavizada / suavizada.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class DistillationStrategy(TrainingStrategy):
    """105 rotulos verdadeiros mais os pseudo-rotulos do professor, com tratamento."""

    required_sources = (LABELED, UNLABELED)
    #: o pool de 6.505 define a epoca; os 105 rotulos sao ciclados. Mesmo
    #: cronograma da trilha B, para que a comparacao isole a perda
    primary_source = UNLABELED

    def __init__(
        self,
        *,
        cache: TeacherCache,
        num_classes: int,
        mode: str = "soft",
        tau: float = DEFAULT_TAU,
        temperature: float = DEFAULT_TEMPERATURE,
        lambda_u: float = DEFAULT_LAMBDA_U,
    ) -> None:
        if mode not in MODES:
            raise DistillationError(f"mode precisa ser um de {MODES}, veio {mode!r}")
        if not 0.0 < tau <= 1.0:
            raise DistillationError(f"tau precisa estar em (0, 1], veio {tau}")
        if lambda_u < 0:
            raise DistillationError(f"lambda_u nao pode ser negativo, veio {lambda_u}")
        if cache.num_classes != num_classes:
            raise DistillationError(
                f"o professor tem {cache.num_classes} classes e o aluno "
                f"{num_classes}; o cache e de outro conjunto"
            )

        self.num_classes = num_classes
        self.mode = mode
        self.tau = tau
        self.temperature = temperature
        self.lambda_u = lambda_u
        self._supervised_loss = nn.CrossEntropyLoss()

        # o cache vira tensores uma vez; buscar por id a cada passo em dict seria
        # Python puro no caminho quente
        self._row = cache.index()
        self._probabilities = torch.from_numpy(
            np.ascontiguousarray(cache.probabilities)
        )
        self._teacher = cache.teacher

    def describe(self) -> str:
        detalhe = {
            "hard": "",
            "filter": f", tau={self.tau}",
            "soft": f", T={self.temperature}",
        }
        return f"destilacao de {self._teacher} (mode={self.mode}{detalhe[self.mode]})"

    # ------------------------------------------------------------------ lookup
    def _teacher_rows(self, image_ids: Any, device: Any) -> torch.Tensor:
        """Probabilidades do professor para estes ids, na ordem do batch."""
        try:
            indices = [self._row[str(i)] for i in image_ids]
        except KeyError as erro:
            raise DistillationError(
                f"imagem {erro.args[0]!r} nao esta no cache do professor. "
                f"O cache cobre {len(self._row)} imagens; rode "
                f"`python scripts/teacher.py` sobre o mesmo conjunto que o aluno treina."
            ) from None
        return self._probabilities[indices].to(device)

    # -------------------------------------------------------------------- passo
    def training_step(
        self, model: "_nn.Module", batch: Batch, *, device: Any
    ) -> StepOutput:
        images, labels = batch[LABELED]
        pool, image_ids = batch[UNLABELED]
        if isinstance(pool, (list, tuple)):
            raise DistillationError(
                "a destilacao espera uma vista por imagem, e o loader entregou "
                f"{len(pool)}. Use `augment_unlabeled: weak` (ou `basic`), nao "
                "`weak_strong`: o par fraco/forte e da trilha B."
            )
        images, labels = images.to(device), labels.to(device)
        pool = pool.to(device)

        # um forward sobre os dois ramos concatenados, pelo mesmo motivo da D15:
        # com forwards separados a BatchNorm atualizaria as estatisticas duas vezes
        # por passo, sob distribuicoes diferentes
        n_rot = images.size(0)
        logits = model(torch.cat([images, pool]))
        logits_labeled, logits_pool = logits[:n_rot], logits[n_rot:]

        supervised = self._supervised_loss(logits_labeled, labels)
        professor = self._teacher_rows(image_ids, device)
        confidence, duro = professor.max(dim=1)

        if self.mode == "soft":
            alvo = soften(professor, self.temperature)
            # KL com alvo fixo: difere da CE por uma constante (a entropia do
            # professor), logo mesmo gradiente, mas o numero reportado e zero
            # quando o aluno iguala o professor, o que e mais legivel no log
            destilacao = F.kl_div(
                F.log_softmax(logits_pool / self.temperature, dim=1),
                alvo,
                reduction="batchmean",
            )
            # T^2 mantem a magnitude do gradiente comparavel entre temperaturas
            # (pratica padrao de destilacao, fora do material da disciplina)
            destilacao = destilacao * (self.temperature**2)
            usados = float(professor.size(0))
        else:
            por_imagem = F.cross_entropy(logits_pool, duro, reduction="none")
            if self.mode == "filter":
                mask = (confidence >= self.tau).float()
                # denominador fixo, como na trilha B: dividir pelos que passaram
                # faria o peso do termo crescer conforme o filtro aperta
                destilacao = (por_imagem * mask).mean()
                usados = float(mask.sum())
            else:
                destilacao = por_imagem.mean()
                usados = float(professor.size(0))

        loss = supervised + self.lambda_u * destilacao
        return StepOutput(
            loss=loss,
            components={
                "ce": float(supervised.detach()),
                "distill": float(destilacao.detach()),
                # fracao do lote que o tratamento manteve: 1,0 em hard e soft,
                # menor em filter. E o numero que mostra o custo do filtro
                "kept_rate": usados / max(professor.size(0), 1),
                # confianca media do professor neste lote — nao muda com o treino,
                # mas documenta o regime de ruido em que o aluno esta aprendendo
                "teacher_confidence": float(confidence.mean().detach()),
            },
        )


@register_strategy("distillation")
def build_distillation(
    *,
    num_classes: int,
    cache: str | Path | TeacherCache | None = None,
    mode: str = "soft",
    tau: float = DEFAULT_TAU,
    temperature: float = DEFAULT_TEMPERATURE,
    lambda_u: float = DEFAULT_LAMBDA_U,
    **_: object,
) -> TrainingStrategy:
    """Constroi o aluno a partir do YAML.

    ``cache`` e o caminho do ``.npz`` que ``scripts/teacher.py`` gravou. Receber um
    caminho e nao um modelo e o ponto: o professor nao entra no processo de treino.

    Ele e declarado opcional na assinatura para que a ausencia produza uma mensagem
    util em vez de um ``TypeError`` de Python: quem erra isto esta escrevendo um YAML,
    nao chamando a funcao.
    """
    if cache is None:
        raise DistillationError(
            "a destilacao precisa de `strategy.params.cache`: o caminho do .npz "
            "que o professor gravou. Rode `python scripts/teacher.py` e aponte "
            "para `outputs/teacher/<professor>_train.npz`."
        )
    if not isinstance(cache, TeacherCache):
        cache = TeacherCache.load(cache)
    return DistillationStrategy(
        cache=cache,
        num_classes=num_classes,
        mode=mode,
        tau=tau,
        temperature=temperature,
        lambda_u=lambda_u,
    )
