"""Roda o professor da trilha C e grava as probabilidades em disco.

    python scripts/teacher.py                       # os dois professores, os dois conjuntos
    python scripts/teacher.py --teacher bioclip     # so um
    python scripts/teacher.py --splits test_public   # so medir, sem rotular o train

Produz, para cada professor:

    outputs/teacher/<professor>_train.npz        pseudo-rotulos das 6.505 (o aluno le)
    outputs/teacher/<professor>_test_public.npz  para medir o professor
    outputs/teacher/zero_shot.md                 F1 zero-shot — obrigatorio 1

**Este script e a fronteira.** Depois dele, o professor sai de cena: o aluno
(`configs/p1c_*.yaml`) le o `.npz` e nunca toca num foundation model. E por isso que
"o professor nao pode fazer parte do modelo entregue" nao depende de disciplina.

**Integridade.** O professor so ve o `train` (para rotular) e o `test_public` (para
medirmos ele, como o enunciado permite explicitamente). O `val` e recusado por
`src/tracks/c/teacher.py:ensure_teacher_may_see`, com excecao, nao com comentario.
O template do prompt e **fixo**: nao escolhemos entre templates, porque escolher
exigiria o `val` (proibido ao professor) ou o `test_public` (proibido para escolha).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.catalog import DataCatalog
from src.data.dataset import UnlabeledImageDataset
from src.data.splits import Split
from src.errors import AppError
from src.metrics import classification_report
from src.tracks.c.teacher import (
    PROMPT_TEMPLATE,
    TEACHER_ALLOWED_SPLITS,
    TeacherCache,
    available_teachers,
    build_zero_shot_head,
    cache_path,
    class_prompts,
    ensure_teacher_may_see,
    iter_batches,
    load_teacher,
    resolve_teacher,
    teacher_probabilities,
)
from src.tracking import describe_device, resolve_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAIDA = PROJECT_ROOT / "outputs" / "teacher"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Professor zero-shot da trilha C")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--saida", type=Path, default=SAIDA)
    parser.add_argument(
        "--teacher",
        nargs="+",
        default=list(available_teachers()),
        choices=list(available_teachers()),
        help="quais professores rodar (padrao: todos)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=[s.value for s in TEACHER_ALLOWED_SPLITS],
        help=f"conjuntos a rotular (permitidos: {[s.value for s in TEACHER_ALLOWED_SPLITS]})",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def loader_de(
    catalog: DataCatalog, split: Split, preprocess, *, batch_size: int, num_workers: int
) -> DataLoader:
    """Todas as imagens do conjunto, com o pre-processamento **do professor**.

    Sem rotulo de proposito, mesmo no ``test_public``: o professor nunca recebe
    rotulo nosso. Os rotulos entram depois, so para calcular o F1 dele, casados por
    ``image_id``.
    """
    dataset = UnlabeledImageDataset.from_catalog(catalog, split, transform=preprocess)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def zero_shot_report(cache: TeacherCache, catalog: DataCatalog, split: Split):
    """F1 zero-shot do professor — o obrigatorio 1.

    Casa pseudo-rotulo e verdade por ``image_id``, nao por posicao: o loader e o CSV
    de rotulos nao precisam estar na mesma ordem, e confiar na ordem e o tipo de bug
    que produz um numero plausivel e errado.
    """
    verdade = {ex.image_id: ex.label_index for ex in catalog.labeled(split)}
    linha = cache.index()
    ids = [i for i in cache.image_ids if i in verdade]
    if not ids:
        raise AppError(f"nenhum id do cache casa com os rotulos de {split.value}")
    y_true = np.array([verdade[i] for i in ids], dtype=np.int64)
    y_pred = cache.probabilities[[linha[i] for i in ids]].argmax(axis=1)
    return classification_report(y_true, y_pred, num_classes=cache.num_classes), ids


def tabela_markdown(resultados: dict[str, object], classes: tuple[str, ...]) -> str:
    linhas = [
        "# Trilha C — F1 zero-shot do professor (obrigatorio 1)",
        "",
        f"Medido no `test_public`, com os nomes das classes como prompt: "
        f"`{PROMPT_TEMPLATE}`.",
        "O template e **fixo de antemao**: escolher entre templates seria selecao de",
        "modelo, e o professor nao pode ver o `val`, enquanto o `test_public` e",
        "proibido para escolha. Os dois professores sao ViT-B/16 — a diferenca mede o",
        "pre-treino, nao o tamanho.",
        "",
        "| Professor | Pre-treino | F1 macro | Acuracia |",
        "|---|---|---|---|",
    ]
    for nome, (relatorio, descricao) in sorted(resultados.items()):
        linhas.append(
            f"| `{nome}` | {descricao} | {relatorio.macro_f1:.3f} | "
            f"{relatorio.accuracy:.3f} |"
        )
    linhas += [
        "",
        "## F1 por classe",
        "",
        "| Classe | Suporte | "
        + " | ".join(f"`{n}`" for n in sorted(resultados))
        + " |",
        "|---|---:|" + "---|" * len(resultados),
    ]
    primeiro = next(iter(resultados.values()))[0]
    for indice, classe in enumerate(classes):
        celulas = [
            f"{resultados[n][0].per_class_f1[indice]:.3f}" for n in sorted(resultados)
        ]
        linhas.append(
            f"| {classe} | {primeiro.support[indice]} | " + " | ".join(celulas) + " |"
        )
    return "\n".join(linhas) + "\n"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    catalog = DataCatalog.open(args.data_root)
    classes = catalog.classes.labels

    splits = [ensure_teacher_may_see(Split(s)) for s in args.splits]
    args.saida.mkdir(parents=True, exist_ok=True)
    print(f"dispositivo: {describe_device(device)}")
    print(f"prompt fixo: {PROMPT_TEMPLATE}")
    print(f"exemplo:     {class_prompts(classes)[0]!r}\n")

    resultados: dict[str, object] = {}
    for nome in args.teacher:
        spec = resolve_teacher(nome)
        print(f"##### {nome} — {spec.describe()}")
        model, tokenizer, preprocess = load_teacher(spec, device=device)
        head = build_zero_shot_head(model, tokenizer, classes, device=device)

        for split in splits:
            loader = loader_de(
                catalog,
                split,
                preprocess,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            probabilidades, ids = teacher_probabilities(
                model, head, iter_batches(loader), device=device
            )
            cache = TeacherCache(
                teacher=nome,
                split=split.value,
                image_ids=ids,
                probabilities=probabilidades,
                class_names=tuple(classes),
            )
            caminho = cache.save(cache_path(args.saida, nome, split))
            print(f"  {split.value:12} {len(ids):5d} imagens -> {caminho.name}")

            if split is Split.TEST_PUBLIC:
                relatorio, casados = zero_shot_report(cache, catalog, split)
                resultados[nome] = (relatorio, spec.describe())
                print(
                    f"  {'zero-shot':12} F1 macro {relatorio.macro_f1:.4f} | "
                    f"acuracia {relatorio.accuracy:.4f}  ({len(casados)} rotulos)"
                )
        del model, head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if resultados:
        tabela = args.saida / "zero_shot.md"
        tabela.write_text(tabela_markdown(resultados, classes), encoding="utf-8")
        print(f"\n  tabela: {tabela}")


if __name__ == "__main__":
    try:
        main()
    except AppError as erro:
        raise SystemExit(f"erro: {erro}")
