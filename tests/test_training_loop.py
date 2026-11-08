"""Integracao do encanamento: laco de treino, avaliacao, checkpoint e resume.

Usa um modelo e uma estrategia falsos, definidos aqui e nao registrados no
pacote: o objetivo e verificar a estrutura, nao uma decisao de modelagem. Quando
o backbone e a estrategia de verdade existirem, nada deste arquivo precisa
mudar — e isso que mostra que o laco nao depende deles.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.loaders import LABELED, UNLABELED
from src.data.splits import Split
from src.evaluation import best_epoch, evaluate
from src.methods.base import StepOutput, TrainingStrategy
from src.models.checkpoint import BEST_NAME, LAST_NAME, load_checkpoint
from src.training import (
    CheckpointWriter,
    ConsoleLogger,
    HistoryWriter,
    Trainer,
    TrainerError,
    build_optimizer,
)

NUM_CLASSES = 21
FEATURES = 8


class FakeModel(nn.Module):
    """Linear sobre vetores — rapido e suficiente para exercitar o laco."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.fc = nn.Linear(FEATURES, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class FakeStrategy(TrainingStrategy):
    """Entropia cruzada sobre a fonte rotulada."""

    required_sources = (LABELED,)
    primary_source = LABELED

    def __init__(self) -> None:
        self.epocas_vistas: list[int] = []
        self._loss = nn.CrossEntropyLoss()

    def on_epoch_start(self, epoch: int) -> None:
        self.epocas_vistas.append(epoch)

    def training_step(self, model, batch, *, device) -> StepOutput:
        images, labels = batch[LABELED]
        logits = model(images.to(device))
        loss = self._loss(logits, labels.to(device))
        return StepOutput(loss=loss, components={"ce": float(loss.detach())})


class FakeStrategyWithUnlabeled(FakeStrategy):
    required_sources = (LABELED, UNLABELED)
    primary_source = LABELED

    def training_step(self, model, batch, *, device) -> StepOutput:
        assert UNLABELED in batch, "o laco precisa entregar as duas fontes"
        imagens_sem_rotulo, ids = batch[UNLABELED]
        assert len(ids) > 0
        return super().training_step(model, batch, device=device)


class FakeLabeledDataset:
    """Vetores com rotulo, deterministico pela seed."""

    def __init__(self, n: int, *, seed: int = 0) -> None:
        gerador = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, FEATURES, generator=gerador)
        self.y = torch.randint(0, NUM_CLASSES, (n,), generator=gerador)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], int(self.y[i])


class FakeUnlabeledDataset(FakeLabeledDataset):
    def __getitem__(self, i: int):
        return self.x[i], f"img{i:04d}"


@pytest.fixture
def pecas(tmp_path):
    torch.manual_seed(0)
    modelo = FakeModel()
    strategy = FakeStrategy()
    return {
        "model": modelo,
        "strategy": strategy,
        "optimizer": build_optimizer(modelo, lr=0.05),
        "loaders": {
            LABELED: DataLoader(FakeLabeledDataset(32), batch_size=8, shuffle=False)
        },
        "validation_loader": DataLoader(FakeLabeledDataset(16, seed=1), batch_size=8),
        "run_dir": tmp_path / "run",
        "num_classes": NUM_CLASSES,
    }


def test_treino_produz_historico_e_artefatos(pecas):
    trainer = Trainer(
        **pecas,
        epochs=3,
        observers=(ConsoleLogger(), HistoryWriter(), CheckpointWriter()),
    )
    historico = trainer.fit()

    assert [r.epoch for r in historico] == [1, 2, 3]
    assert all(r.train_loss > 0 for r in historico)
    assert pecas["strategy"].epocas_vistas == [1, 2, 3]

    run_dir = pecas["run_dir"]
    assert (run_dir / LAST_NAME).exists()
    assert (run_dir / BEST_NAME).exists()
    historico_json = json.loads((run_dir / "history.json").read_text())
    assert [row["epoch"] for row in historico_json] == [1, 2, 3]
    assert "val_macro_f1" in historico_json[0]
    assert "loss_ce" in historico_json[0]


def test_validacao_de_cada_epoca_vem_do_val(pecas):
    historico = Trainer(**pecas, epochs=2).fit()
    assert all(r.validation is not None for r in historico)
    assert all(r.validation.split is Split.VAL for r in historico)
    # o historico de validacao pode ser usado para selecao justamente por isso
    assert best_epoch([r.validation for r in historico]) in (1, 2)


def test_best_pt_guarda_a_metrica_monitorada(pecas):
    Trainer(**pecas, epochs=3, observers=(CheckpointWriter(),)).fit()
    estado = load_checkpoint(pecas["run_dir"] / BEST_NAME, model=FakeModel())
    assert estado.monitor == "macro_f1"
    assert estado.best_metric > float("-inf")


def test_resume_continua_da_epoca_seguinte(pecas, capsys):
    Trainer(**pecas, epochs=2, observers=(CheckpointWriter(),)).fit()

    modelo = FakeModel()
    strategy = FakeStrategy()
    segundo = Trainer(
        model=modelo,
        strategy=strategy,
        optimizer=build_optimizer(modelo, lr=0.05),
        loaders=pecas["loaders"],
        validation_loader=pecas["validation_loader"],
        run_dir=pecas["run_dir"],
        num_classes=NUM_CLASSES,
        epochs=4,
        observers=(CheckpointWriter(),),
    )
    historico = segundo.fit(resume=True)
    assert [r.epoch for r in historico] == [3, 4]
    assert "retomando" in capsys.readouterr().out


def test_resume_sem_checkpoint_comeca_do_zero(pecas, capsys):
    historico = Trainer(**pecas, epochs=1).fit(resume=True)
    assert [r.epoch for r in historico] == [1]
    assert "comecando do zero" in capsys.readouterr().out


def test_fonte_ausente_e_recusada_na_montagem(pecas):
    pecas["strategy"] = FakeStrategyWithUnlabeled()
    with pytest.raises(TrainerError, match="unlabeled"):
        Trainer(**pecas, epochs=1)


def test_metodo_com_duas_fontes_recebe_as_duas(pecas):
    """O pool sem rotulo e ciclado: a epoca tem o tamanho da fonte primaria.

    E o que o FixMatch precisa — 105 rotulos contra 6.505 imagens sem rotulo nao
    pode fazer a epoca terminar em 4 batches.
    """
    pecas["strategy"] = FakeStrategyWithUnlabeled()
    pecas["loaders"] = {
        **pecas["loaders"],
        UNLABELED: DataLoader(FakeUnlabeledDataset(8), batch_size=8, shuffle=False),
    }
    historico = Trainer(**pecas, epochs=1).fit()
    assert len(historico) == 1


def test_loader_vazio_falha_com_mensagem_clara(pecas):
    pecas["loaders"] = {LABELED: DataLoader(FakeLabeledDataset(0), batch_size=8)}
    with pytest.raises(TrainerError, match="loader vazio"):
        Trainer(**pecas, epochs=1).fit()


def test_avaliacao_recusa_logits_de_forma_errada(pecas):
    class ModeloErrado(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 3)

    with pytest.raises(ValueError, match="esperado \\(batch, 21\\)"):
        evaluate(
            ModeloErrado(),
            pecas["validation_loader"],
            split=Split.VAL,
            num_classes=NUM_CLASSES,
        )


def test_avaliacao_devolve_confianca_por_imagem(pecas):
    resultado = evaluate(
        FakeModel(),
        pecas["validation_loader"],
        split=Split.VAL,
        num_classes=NUM_CLASSES,
    )
    assert len(resultado.confidence) == 16
    assert len(resultado.y_true) == len(resultado.y_pred) == 16
    assert all(1 / NUM_CLASSES <= c <= 1.0 for c in resultado.confidence)


def test_avaliacao_restaura_o_modo_do_modelo(pecas):
    modelo = FakeModel()
    modelo.train()
    evaluate(
        modelo, pecas["validation_loader"], split=Split.VAL, num_classes=NUM_CLASSES
    )
    assert modelo.training is True
