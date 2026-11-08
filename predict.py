"""Interface de entrega exigida pelo enunciado.

    python predict.py --imagens PASTA --saida preds.csv

Escreve ``image_id,label`` usando os nomes de ``classes.csv``, roda em CPU, carrega
somente pesos locais e imprime a contagem de parametros do modelo.

Restricoes que este arquivo garante:
  * CPU: ``map_location="cpu"`` e ``torch.device("cpu")``, sem tocar em CUDA;
  * sem internet: o backbone e construido com ``pretrained=False`` e os pesos vem
    do checkpoint em disco, nunca de download;
  * teto de 12M parametros, verificado na construcao.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import replace  # noqa: E402

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.augment import build_augmentation  # noqa: E402
from src.data.catalog import ClassVocabulary  # noqa: E402
from src.data.dataset import UnlabeledImageDataset  # noqa: E402
from src.errors import AppError  # noqa: E402
from src.evaluation import predict_labels  # noqa: E402
from src.models import build_model, count_parameters, load_checkpoint  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "entrega.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "modelo.pt"
DEFAULT_CLASSES = PROJECT_ROOT / "data" / "classes.csv"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classifica as imagens de uma pasta e escreve image_id,label"
    )
    parser.add_argument("--imagens", required=True, type=Path, help="pasta de imagens")
    parser.add_argument("--saida", required=True, type=Path, help="CSV de saida")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pesos", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def find_images(folder: Path) -> tuple[Path, ...]:
    if not folder.is_dir():
        raise SystemExit(f"pasta de imagens nao encontrada: {folder}")
    paths = sorted(p for p in folder.iterdir() if p.suffix in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"nenhuma imagem em {folder}")
    return tuple(paths)


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    classes = ClassVocabulary.from_csv(args.classes)
    config = load_config(args.config)
    # pesos vem do checkpoint local: nunca baixar nada na inferencia
    model = build_model(
        replace(config.model, pretrained=False), num_classes=len(classes)
    ).to(device)
    load_checkpoint(args.pesos, model=model, map_location="cpu")
    model.eval()

    print(f"parametros: {count_parameters(model):,}")

    paths = find_images(args.imagens)
    dataset = UnlabeledImageDataset(
        paths=paths,
        image_ids=tuple(p.stem for p in paths),
        transform=build_augmentation("eval", image_size=config.data.image_size),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    predictions, _ = predict_labels(
        model, (images for images, _ in loader), device=device
    )

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    with args.saida.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "label"])
        for image_id, index in zip(dataset.image_ids, predictions):
            writer.writerow([image_id, classes.label(index)])

    print(f"{len(predictions)} imagens -> {args.saida}")


if __name__ == "__main__":
    try:
        main()
    except AppError as erro:
        raise SystemExit(f"erro: {erro}")
