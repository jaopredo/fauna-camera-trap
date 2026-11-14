"""Treina um experimento. Unico lugar onde os componentes sao compostos.

python scripts/train.py --config configs/exemplo.yaml
python scripts/train.py --config configs/exemplo.yaml --set seed=1 data.budget=20
python scripts/train.py --config configs/exemplo.yaml --resume
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  (ajusta sys.path antes de importar src)

from src.config import load_config
from src.data.augment import build_augmentation
from src.data.catalog import DataCatalog
from src.data.loaders import LABELED, labeled_loader, unlabeled_loader
from src.data.splits import Split
from src.errors import AppError
from src.methods import build_strategy
from src.models import build_model
from src.tracking import create_run, describe_device, resolve_device, seed_everything
from src.training import (
    CheckpointWriter,
    ConsoleLogger,
    HistoryWriter,
    Trainer,
    build_optimizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino de um experimento do PA3")
    parser.add_argument(
        "--config", required=True, help="caminho do YAML do experimento"
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=(),
        metavar="CHAVE=VALOR",
        help="sobrescreve chaves do YAML (ex.: data.budget=20 seeds=[0,42])",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="roda so esta seed, que precisa estar declarada em seeds "
        "(util para retomar uma execucao solta sem repetir as outras)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="retoma de <run_dir>/last.pt"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu ou cuda (padrao: auto — usa a GPU se houver). "
        "Com ROCm o dispositivo tambem se chama cuda.",
    )
    return parser.parse_args()


def treinar_uma_seed(config, seed: int, args) -> None:
    """Uma execucao: uma seed, um diretorio, um checkpoint.

    Tudo e reconstruido aqui dentro — pesos, embaralhamento, otimizador. Reaproveitar
    o modelo entre seeds faria a segunda execucao partir da primeira, e as tres
    deixariam de ser amostras independentes do mesmo experimento.
    """
    seed_everything(seed)
    run = create_run(
        config.output_dir,
        name=config.run_name(seed),
        seed=seed,
        config=config.as_dict(),
        device=args.device,
    )

    catalog = DataCatalog.open(config.data.root)
    num_classes = len(catalog.classes)
    transform = build_augmentation(
        config.data.augment, image_size=config.data.image_size
    )

    model = build_model(config.model, num_classes=num_classes).to(args.device)
    strategy = build_strategy(config.strategy, num_classes=num_classes)

    loaders = {
        LABELED: labeled_loader(
            catalog, Split.TRAIN, config.data, transform=transform, shuffle=True
        )
    }
    if "unlabeled" in strategy.required_sources:
        loaders["unlabeled"] = unlabeled_loader(
            catalog, config.data, transform=transform
        )

    # a validacao usa a politica de avaliacao, nunca a de treino
    validation = labeled_loader(
        catalog,
        Split.VAL,
        config.data,
        transform=build_augmentation("eval", image_size=config.data.image_size),
        shuffle=False,
    )

    trainer = Trainer(
        model=model,
        strategy=strategy,
        optimizer=build_optimizer(
            model, lr=config.optim.lr, weight_decay=config.optim.weight_decay
        ),
        loaders=loaders,
        validation_loader=validation,
        run_dir=run.run_dir,
        num_classes=num_classes,
        epochs=config.optim.epochs,
        monitor=config.optim.monitor,
        observers=(
            ConsoleLogger(),
            HistoryWriter(),
            CheckpointWriter(save_every=config.optim.save_every),
        ),
        device=args.device,
        config=config.as_dict(),
    )
    trainer.fit(resume=args.resume)


def main() -> None:
    args = parse_args()
    config = load_config(args.config, overrides=args.set)

    if args.seed is not None:
        if args.seed not in config.seeds:
            raise AppError(
                f"--seed {args.seed} nao esta declarada no config: "
                f"seeds={list(config.seeds)}. Acrescente-a ao YAML ou use "
                f"--set seeds=[{args.seed}]."
            )
        seeds = (args.seed,)
    else:
        seeds = config.seeds

    args.device = resolve_device(args.device)
    print(
        f"{config.name}: {len(seeds)} execucao(oes), seeds {list(seeds)}\n"
        f"dispositivo: {describe_device(args.device)}\n",
        flush=True,
    )
    for posicao, seed in enumerate(seeds, start=1):
        print(f"===== seed {seed} ({posicao}/{len(seeds)}) =====", flush=True)
        treinar_uma_seed(config, seed, args)


if __name__ == "__main__":
    # erro previsto (config errada, pacote ausente, backbone inexistente) sai com
    # uma linha; erro nao previsto continua subindo com traceback, para depurar
    try:
        main()
    except AppError as erro:
        raise SystemExit(f"erro: {erro}")
