"""Avalia um checkpoint num conjunto rotulado.

    python scripts/evaluate.py --config configs/exemplo.yaml --checkpoint outputs/.../best.pt
    python scripts/evaluate.py --config ... --checkpoint ... --split test_public

``--split val`` serve a selecao de modelo; ``--split test_public`` so reporta. A
barreira esta no codigo: ``src.evaluation.selection`` recusa resultado de teste.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src.config import load_config
from src.data.augment import build_augmentation
from src.data.catalog import DataCatalog
from src.data.loaders import labeled_loader
from src.data.splits import Split, parse_split
from src.errors import AppError
from src.evaluation import evaluate
from src.models import build_model, load_checkpoint
from src.tracking import resolve_device, seed_everything, write_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avaliacao de um checkpoint do PA3")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--split",
        default=Split.VAL.value,
        choices=[Split.VAL.value, Split.TEST_PUBLIC.value],
        help="val para selecao; test_public apenas para reportar",
    )
    parser.add_argument("--set", nargs="*", default=(), metavar="CHAVE=VALOR")
    parser.add_argument("--device", default="auto", help="auto, cpu ou cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, overrides=args.set)
    # a avaliacao e deterministica; fixar a seed so mantem o ambiente igual ao
    # do treino, caso alguma transformacao futura use aleatoriedade
    seed_everything(config.seeds[0])
    split = parse_split(args.split)
    args.device = resolve_device(args.device)

    catalog = DataCatalog.open(config.data.root)
    num_classes = len(catalog.classes)
    model = build_model(config.model, num_classes=num_classes).to(args.device)
    load_checkpoint(args.checkpoint, model=model, map_location=args.device)

    loader = labeled_loader(
        catalog,
        split,
        config.data,
        transform=build_augmentation("eval", image_size=config.data.image_size),
        shuffle=False,
    )
    result = evaluate(
        model, loader, split=split, num_classes=num_classes, device=args.device
    )

    print(f"{split.value}: {len(loader.dataset)} imagens")
    print(f"  F1 macro  {result.macro_f1:.4f}")
    print(f"  acuracia  {result.accuracy:.4f}")
    print("\n  F1 por classe:")
    for index, value in enumerate(result.report.per_class_f1):
        support = result.report.support[index]
        print(f"    {catalog.classes.label(index):24s} {value:.4f}  (n={support})")

    destination = write_metrics(
        args.checkpoint.parent,
        f"metrics_{split.value}",
        {"split": split.value, **result.report.as_dict()},
    )
    print(f"\n  metricas: {destination}")


if __name__ == "__main__":
    # erro previsto (config errada, pacote ausente, backbone inexistente) sai com
    # uma linha; erro nao previsto continua subindo com traceback, para depurar
    try:
        main()
    except AppError as erro:
        raise SystemExit(f"erro: {erro}")
