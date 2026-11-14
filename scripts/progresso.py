"""Acompanha as execucoes sem esperar o fim.

Le o ``history.json`` de cada diretorio em ``outputs/`` e mostra em que epoca cada
execucao esta e qual a melhor metrica de validacao ate agora.

Existe porque o treino e longo e roda em background: o ``history.json`` e reescrito
a cada epoca, entao da para acompanhar sem interromper nada.

    python scripts/progresso.py
    python scripts/progresso.py --outputs outputs --monitor macro_f1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Progresso das execucoes do PA3")
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--monitor", default="macro_f1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chave = f"val_{args.monitor}"

    execucoes = sorted(
        d for d in args.outputs.glob("*/") if (d / "history.json").exists()
    )
    if not execucoes:
        print(f"nenhuma execucao com history.json em {args.outputs}")
        return

    for run_dir in execucoes:
        historico = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        if not historico:
            continue
        ultima = historico[-1]
        melhor = max(historico, key=lambda linha: linha.get(chave, -1.0))
        total = sum(linha.get("seconds", 0.0) for linha in historico)
        print(
            f"{run_dir.name:26} epoca {ultima['epoch']:>3} | "
            f"treino {ultima['train_loss']:.4f} | "
            f"melhor {args.monitor} {melhor.get(chave, 0.0):.4f} "
            f"(epoca {melhor['epoch']:>3}) | {total / 60:.1f} min"
        )


if __name__ == "__main__":
    main()
