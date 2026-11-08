"""Laco de treino, independente do metodo.

Responsabilidade unica: percorrer epocas e batches, chamar o metodo, aplicar o
otimizador, avaliar na validacao e emitir eventos. Nao conhece perda, arquitetura
nem trilha — por isso trocar de metodo nunca exige editar este arquivo.

Composicao e injecao ficam nos ``scripts/``: o ``Trainer`` recebe modelo, metodo,
otimizador, loaders e observadores prontos.
"""

from __future__ import annotations

import time
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

import torch

from src.data.splits import Split
from src.errors import AppError
from src.evaluation.evaluator import evaluate
from src.methods.base import TrainingStrategy
from src.models.checkpoint import LAST_NAME, load_checkpoint
from src.training.observers import EpochRecord, TrainingContext, TrainingObserver

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader


class TrainerError(AppError, RuntimeError):
    """Montagem invalida do treino."""


class Trainer:
    """Executa o treino de um metodo sobre um modelo."""

    def __init__(
        self,
        *,
        model: "nn.Module",
        strategy: TrainingStrategy,
        optimizer: "Optimizer",
        loaders: Mapping[str, "DataLoader"],
        run_dir: Path,
        num_classes: int,
        epochs: int,
        monitor: str = "macro_f1",
        validation_loader: "DataLoader | None" = None,
        observers: Sequence[TrainingObserver] = (),
        device: Any = "cpu",
        config: Mapping[str, Any] | None = None,
    ) -> None:
        missing = set(strategy.required_sources) - set(loaders)
        if missing:
            raise TrainerError(
                f"{type(strategy).__name__} precisa das fontes {sorted(missing)}, "
                f"que nao foram fornecidas. Recebidas: {sorted(loaders)}"
            )
        if strategy.primary_source not in loaders:
            raise TrainerError(
                f"fonte primaria {strategy.primary_source!r} ausente dos loaders "
                f"{sorted(loaders)}"
            )
        self._model = model
        self._strategy = strategy
        self._optimizer = optimizer
        self._loaders = dict(loaders)
        self._validation_loader = validation_loader
        self._observers = tuple(observers)
        self._device = device
        self._num_classes = num_classes
        self._epochs = epochs
        self._monitor = monitor
        self._run_dir = run_dir
        self._config = config

    @property
    def context(self) -> TrainingContext:
        return TrainingContext(
            model=self._model,
            strategy=self._strategy,
            optimizer=self._optimizer,
            run_dir=self._run_dir,
            monitor=self._monitor,
            total_epochs=self._epochs,
            config=self._config,
        )

    def fit(self, *, resume: bool = False) -> tuple[EpochRecord, ...]:
        """Treina e devolve o historico. Com ``resume``, retoma de ``last.pt``."""
        context = self.context
        first_epoch = 1
        if resume:
            first_epoch = self._resume()
        for observer in self._observers:
            observer.on_train_start(context)

        history: list[EpochRecord] = []
        for epoch in range(first_epoch, self._epochs + 1):
            started = time.perf_counter()
            self._strategy.on_epoch_start(epoch)
            train_loss, components = self._train_one_epoch()
            validation = None
            if self._validation_loader is not None:
                validation = evaluate(
                    self._model,
                    self._validation_loader,
                    split=Split.VAL,
                    num_classes=self._num_classes,
                    device=self._device,
                )
            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                seconds=time.perf_counter() - started,
                components=components,
                validation=validation,
            )
            history.append(record)
            for observer in self._observers:
                observer.on_epoch_end(record, context)

        for observer in self._observers:
            observer.on_train_end(tuple(history), context)
        return tuple(history)

    def _train_one_epoch(self) -> tuple[float, dict[str, float]]:
        self._model.train()
        total_loss = 0.0
        totals: dict[str, float] = {}
        batches = 0
        for batch in self._iterate_batches():
            self._optimizer.zero_grad(set_to_none=True)
            output = self._strategy.training_step(
                self._model, batch, device=self._device
            )
            output.loss.backward()
            self._optimizer.step()
            total_loss += float(output.loss.detach())
            for name, value in output.components.items():
                totals[name] = totals.get(name, 0.0) + float(value)
            batches += 1
        if batches == 0:
            raise TrainerError("a epoca nao produziu nenhum batch: loader vazio?")
        return total_loss / batches, {k: v / batches for k, v in totals.items()}

    def _iterate_batches(self) -> Iterator[dict[str, Any]]:
        """Combina as fontes pedidas pelo metodo.

        A fonte primaria define o tamanho da epoca; as demais sao cicladas. Isso e
        o que o FixMatch precisa: o pool sem rotulo tem 6.505 imagens e o rotulado
        tem 105, e uma epoca nao pode terminar depois de 4 batches.
        """
        primary = self._strategy.primary_source
        secondary = [s for s in self._strategy.required_sources if s != primary]
        companions = {name: cycle(self._loaders[name]) for name in secondary}
        for item in self._loaders[primary]:
            batch = {primary: item}
            for name, iterator in companions.items():
                batch[name] = next(iterator)
            yield batch

    def _resume(self) -> int:
        path = self._run_dir / LAST_NAME
        if not path.exists():
            print(f"--resume pedido, mas {path} nao existe: comecando do zero")
            return 1
        state = load_checkpoint(
            path,
            model=self._model,
            optimizer=self._optimizer,
            map_location=str(self._device),
        )
        self._strategy.load_state_dict(state.strategy_state)
        for observer in self._observers:
            prime = getattr(observer, "prime", None)
            if callable(prime):
                prime(state.best_metric)
        print(
            f"retomando de {path} — epoca {state.epoch}, "
            f"melhor {state.monitor} {state.best_metric:.4f}"
        )
        return state.epoch + 1


def build_optimizer(
    model: "nn.Module", *, lr: float, weight_decay: float = 0.0
) -> "Optimizer":
    """Otimizador padrao do projeto.

    Adam porque e o que os laboratorios usaram; fica isolado aqui para que trocar
    por SGD com momento seja uma linha, se a validacao pedir.
    """
    return torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
