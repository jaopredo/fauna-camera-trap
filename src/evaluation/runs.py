"""Reconstruir uma execucao salva, para avaliar ou diagnosticar depois.

Existe porque tres relatorios precisam do mesmo gesto — "me devolva o modelo que
esta execucao salvou, e as probabilidades dele neste conjunto" — e tres copias da
mesma funcao divergem. Antes desta extracao, ``report_p0.py`` e ``report_p1.py`` ja
tinham cada um a sua versao.

Tambem e o que permite que o relatorio de **cada trilha** seja independente: sem um
lugar comum, o relatorio da trilha C teria de importar do relatorio da trilha B, e
apagar a trilha B quebraria a C. Ver ``tests/test_tracks_independence.py``.

Isto nao e composicao de experimento: nao escolhe modelo, nao escreve nada e nao
decide nada. Le um diretorio e devolve objetos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from src.config.schema import ExperimentConfig
from src.data.augment import build_augmentation
from src.data.catalog import DataCatalog
from src.data.loaders import labeled_loader
from src.data.splits import Split
from src.errors import AppError
from src.models import build_model, count_parameters, load_checkpoint
from src.models.checkpoint import BEST_NAME
from src.tracking import manifest_seed, read_manifest


@dataclass(frozen=True)
class LoadedRun:
    """O que uma execucao salva devolve quando reaberta."""

    run_dir: Path
    config: ExperimentConfig
    seed: int
    model: torch.nn.Module
    catalog: DataCatalog

    @property
    def classes(self) -> tuple[str, ...]:
        return self.catalog.classes.labels

    @property
    def num_classes(self) -> int:
        return len(self.catalog.classes)

    def num_parameters(self) -> int:
        return count_parameters(self.model)


def load_run(run_dir: Path | str, *, device: str) -> LoadedRun:
    """Reconstroi o modelo da execucao com os pesos do ``best.pt``.

    ``pretrained=False`` na reconstrucao: os pesos vem do checkpoint, nunca de
    download. Isso vale aqui tambem, nao so no ``predict.py`` — um relatorio que
    baixasse pesos daria numero de outro modelo.
    """
    run_dir = Path(run_dir)
    manifesto = read_manifest(run_dir)
    config = ExperimentConfig.from_raw(manifesto["config"])
    catalog = DataCatalog.open(config.data.root)
    model = build_model(
        replace(config.model, pretrained=False), num_classes=len(catalog.classes)
    ).to(device)
    load_checkpoint(run_dir / BEST_NAME, model=model, map_location=device)
    return LoadedRun(
        run_dir=run_dir,
        config=config,
        seed=manifest_seed(manifesto),
        model=model,
        catalog=catalog,
    )


def eval_loader(run: LoadedRun, split: Split) -> DataLoader:
    """Loader deterministico daquele conjunto, com a aumentacao ``eval``."""
    return labeled_loader(
        run.catalog,
        split,
        run.config.data,
        transform=build_augmentation("eval", image_size=run.config.data.image_size),
        shuffle=False,
    )


@torch.no_grad()
def probabilities(
    model: torch.nn.Module, loader: DataLoader, *, device: str
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Matriz ``(n, num_classes)`` e os rotulos verdadeiros, na ordem do loader.

    Um forward so: as varreduras de limiar e as curvas por classe saem depois, em
    memoria, sem passar o conjunto pelo modelo de novo.
    """
    model.eval()
    probs, verdadeiros = [], []
    for images, labels in loader:
        probs.append(torch.softmax(model(images.to(device)).float(), dim=1).cpu())
        verdadeiros.append(torch.as_tensor(labels))
    return torch.cat(probs).numpy(), torch.cat(verdadeiros).numpy()


def best_validation_metric(
    run_dir: Path | str, *, monitor: str = "macro_f1"
) -> tuple[float, int]:
    """Melhor valor da metrica de **validacao** desta execucao, e a epoca dele.

    E o numero que escolheu o checkpoint: o ``Trainer`` avalia no ``val`` a cada
    epoca, grava em ``history.json`` sob ``val_<monitor>`` e guarda o ``best.pt`` da
    melhor. Ler daqui e, portanto, reusar a selecao que ja foi feita no ``val`` — nao
    e uma selecao nova.

    **Por que isto nao burla a barreira do ``test_public``.** O ``history.json`` so
    contem metricas de validacao e de treino: o laco de treino nunca toca no conjunto
    de teste, logo nao existe chave de teste para esta funcao retornar por acidente.
    A garantia e estrutural, nao uma convencao — e por isso que esta funcao pode
    viver fora de ``src.evaluation.selection``, que e onde ficam as funcoes que
    recusam conjunto proibido.

    Serve para comparar execucoes **sem carregar modelo e sem GPU**, que e o que os
    relatorios precisam quando a escolha de qual execucao reportar tem de sair do
    ``val`` e nao do teste.
    """
    caminho = Path(run_dir) / "history.json"
    if not caminho.exists():
        raise AppError(f"sem history.json em {run_dir}; a execucao nao terminou?")
    historico = json.loads(caminho.read_text(encoding="utf-8"))
    epocas = (
        historico["epochs"]
        if isinstance(historico, dict) and "epochs" in historico
        else historico
    )
    chave = f"val_{monitor}"
    valores = [
        (float(e[chave]), int(e["epoch"])) for e in epocas if e.get(chave) is not None
    ]
    if not valores:
        raise AppError(
            f"{caminho} nao tem {chave!r}. Chaves disponiveis na primeira epoca: "
            f"{sorted(epocas[0]) if epocas else '(vazio)'}"
        )
    return max(valores)
