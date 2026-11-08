"""Strategy de treino: o contrato que o laco de treino consome.

Este e o eixo em que as tres trilhas divergem. O laco nao sabe qual estrategia
esta rodando: ele pergunta de quais fontes de dados ela precisa e, a cada batch,
qual e a perda. Assim FixMatch (rotulado + nao rotulado, par fraco/forte), SimCLR
(duas vistas, sem rotulo) e professor-aluno (pseudo-rotulos + 105 rotulos) entram
como arquivos novos, sem editar ``src/training/loop.py``.

Toda estrategia cumpre o mesmo contrato, verificado pelo teste comum em
``tests/test_strategy_contract.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from torch import Tensor, nn

#: Um batch e um mapeamento fonte -> o que aquele loader devolveu.
#: Ex.: ``{"labeled": (imagens, rotulos), "unlabeled": (imagens, ids)}``.
Batch = Mapping[str, Any]


@dataclass(frozen=True)
class StepOutput:
    """Resultado de um passo de treino.

    ``components`` guarda as parcelas da perda em float (ja fora do grafo), porque
    com perdas somadas o total sozinho esconde qual parcela estagnou.
    """

    loss: "Tensor"
    components: Mapping[str, float] = field(default_factory=dict)


class TrainingStrategy(ABC):
    """Como transformar um batch em perda.

    Nao possui o modelo nem o otimizador: recebe o modelo no passo e deixa a
    otimizacao para o laco. Isso permite a mesma estrategia rodar sobre backbones
    diferentes (a avaliacao linear da Trilha A exige exatamente isso).
    """

    #: Fontes de dados que esta estrategia consome, pelos nomes de
    #: ``src.data.loaders``.
    required_sources: tuple[str, ...] = ()

    #: Fonte que define o tamanho da epoca; as outras sao cicladas.
    primary_source: str = ""

    @abstractmethod
    def training_step(
        self, model: "nn.Module", batch: Batch, *, device: Any
    ) -> StepOutput:
        """Calcula a perda de um batch. Nao chama ``backward`` nem o otimizador."""

    def on_epoch_start(self, epoch: int) -> None:
        """Gancho por epoca — rampa de peso, limiar por classe, EMA do professor."""

    def state_dict(self) -> dict[str, Any]:
        """Estado proprio da estrategia a guardar no checkpoint (padrao: nenhum)."""
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restaura o estado proprio da estrategia ao retomar um treino."""
