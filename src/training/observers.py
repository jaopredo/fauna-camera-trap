"""Observadores do laco de treino: log, historico e checkpoint.

O laco nao imprime nem grava nada: ele emite eventos. Isso mantem o treino
testavel sem disco e deixa a composicao (quais observadores existem) nos
``scripts/``. Os ganchos tem implementacao vazia na base para que cada observador
sobrescreva so o que precisa.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from src.evaluation.evaluator import EvaluationResult
from src.evaluation.selection import is_improvement
from src.methods.base import TrainingStrategy
from src.models.checkpoint import BEST_NAME, LAST_NAME, CheckpointState, save_checkpoint

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer


@dataclass(frozen=True)
class EpochRecord:
    """O que uma epoca produziu."""

    epoch: int
    train_loss: float
    seconds: float
    components: Mapping[str, float] = field(default_factory=dict)
    validation: EvaluationResult | None = None

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "seconds": self.seconds,
            **{f"loss_{k}": v for k, v in self.components.items()},
        }
        if self.validation is not None:
            row["val_macro_f1"] = self.validation.macro_f1
            row["val_accuracy"] = self.validation.accuracy
        return row


@dataclass(frozen=True)
class TrainingContext:
    """O que os observadores podem inspecionar, sem poder alterar o laco."""

    model: "nn.Module"
    strategy: TrainingStrategy
    optimizer: "Optimizer"
    run_dir: Path
    monitor: str
    total_epochs: int
    config: Mapping[str, Any] | None = None


class TrainingObserver:
    """Base com ganchos vazios."""

    def on_train_start(self, context: TrainingContext) -> None: ...

    def on_epoch_end(self, record: EpochRecord, context: TrainingContext) -> None: ...

    def on_train_end(
        self, history: tuple[EpochRecord, ...], context: TrainingContext
    ) -> None: ...


class ConsoleLogger(TrainingObserver):
    """Uma linha por epoca no terminal."""

    def on_train_start(self, context: TrainingContext) -> None:
        print(f"treino em {context.run_dir} | monitor: {context.monitor}")

    def on_epoch_end(self, record: EpochRecord, context: TrainingContext) -> None:
        parts = [f"epoca {record.epoch:3d}/{context.total_epochs}"]
        parts.append(f"treino {record.train_loss:.4f}")
        if len(record.components) > 1:
            detail = " ".join(f"{k} {v:.4f}" for k, v in record.components.items())
            parts.append(f"({detail})")
        if record.validation is not None:
            parts.append(f"val F1 {record.validation.macro_f1:.4f}")
            parts.append(f"acc {record.validation.accuracy:.4f}")
        parts.append(f"{record.seconds:.1f}s")
        print(" | ".join(parts))


class HistoryWriter(TrainingObserver):
    """Grava ``history.json`` no diretorio da execucao.

    Reescreve a cada epoca: se o treino cair, o historico das epocas anteriores
    continua em disco.
    """

    def __init__(self, filename: str = "history.json") -> None:
        self._filename = filename

    def on_epoch_end(self, record: EpochRecord, context: TrainingContext) -> None:
        self._write(context, [record])

    def on_train_end(
        self, history: tuple[EpochRecord, ...], context: TrainingContext
    ) -> None:
        path = context.run_dir / self._filename
        path.write_text(
            json.dumps([r.as_dict() for r in history], indent=2), encoding="utf-8"
        )

    def _write(self, context: TrainingContext, records: list[EpochRecord]) -> None:
        path = context.run_dir / self._filename
        existing: list[dict[str, Any]] = []
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        known = {row["epoch"] for row in existing}
        existing.extend(r.as_dict() for r in records if r.epoch not in known)
        existing.sort(key=lambda row: row["epoch"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


class CheckpointWriter(TrainingObserver):
    """Grava ``last.pt`` (retomavel) e ``best.pt`` (o escolhido no val).

    A escolha do melhor passa por ``is_improvement``, que recusa um resultado que
    nao venha do ``val``.
    """

    def __init__(self, *, save_every: int = 1) -> None:
        self._save_every = save_every
        self._best = float("-inf")

    @property
    def best_metric(self) -> float:
        return self._best

    def prime(self, best_metric: float) -> None:
        """Informa o melhor valor ja visto ao retomar um treino."""
        self._best = best_metric

    def on_epoch_end(self, record: EpochRecord, context: TrainingContext) -> None:
        state = CheckpointState(
            epoch=record.epoch,
            best_metric=self._best,
            monitor=context.monitor,
            history=(),
            config=context.config,
            strategy_state=context.strategy.state_dict(),
        )
        if record.validation is not None and is_improvement(
            record.validation, self._best, monitor=context.monitor
        ):
            self._best = record.validation.metric(context.monitor)
            save_checkpoint(
                context.run_dir / BEST_NAME,
                model=context.model,
                state=replace(state, best_metric=self._best),
                optimizer=context.optimizer,
            )
        if record.epoch % self._save_every == 0 or record.epoch == context.total_epochs:
            save_checkpoint(
                context.run_dir / LAST_NAME,
                model=context.model,
                state=replace(state, best_metric=self._best),
                optimizer=context.optimizer,
            )
