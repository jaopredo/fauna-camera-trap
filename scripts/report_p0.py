"""Relatorio da Parte 0 — o piso.

Varre as execucoes de ``p0_scratch`` e ``p0_finetune``, avalia cada checkpoint no
``test_public`` e produz o que o enunciado pede:

    outputs/p0/tabela.md            F1 macro e acuracia, media +- desvio sobre as seeds
    outputs/p0/f1_por_classe.csv    F1 por classe das duas versoes
    outputs/p0/confusao_finetune.png  matriz de confusao da versao (b)
    outputs/p0/curva_finetune.png     curva de treino e validacao da versao (b)
    curva_orcamento.csv             linhas do orcamento de 5

Uso:
    python scripts/report_p0.py
    python scripts/report_p0.py --figuras-seed 42

**Integridade experimental.** As figuras saem de uma seed fixada de antemao (a
primeira declarada em ``seeds``), nunca da "melhor seed no test_public" — escolher
pelo teste seria selecao de modelo nele. Este script so reporta: nenhuma funcao de
``src.evaluation.selection`` e chamada aqui.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import _bootstrap  # noqa: F401

from src.analysis.budget import BudgetRow, summarize, update_budget_curve
from src.analysis.figures import plot_confusion_matrix, plot_training_curve
from src.config.schema import ExperimentConfig
from src.data.augment import build_augmentation
from src.data.catalog import DataCatalog
from src.data.loaders import labeled_loader
from src.data.splits import Split
from src.errors import AppError
from src.evaluation import EvaluationResult, evaluate
from src.models import build_model, count_parameters, load_checkpoint
from src.models.checkpoint import BEST_NAME
from src.tracking import manifest_seed, read_manifest, resolve_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAIDA = PROJECT_ROOT / "outputs" / "p0"
CURVA_ORCAMENTO = PROJECT_ROOT / "curva_orcamento.csv"

#: Nome da execucao -> rotulo do metodo na tabela e no CSV.
VERSOES = {
    "p0_scratch": "piso_scratch",
    "p0_finetune": "piso_finetune",
}

#: Teto de referencia do enunciado (ResNet-18 com todos os rotulos do train).
TETO_F1 = 0.686


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relatorio da Parte 0 do PA3")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="diretorio onde estao as execucoes",
    )
    parser.add_argument(
        "--figuras-seed",
        type=int,
        default=None,
        help="seed de onde saem a matriz de confusao e a curva "
        "(padrao: a primeira declarada em seeds)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu ou cuda")
    return parser.parse_args()


def figuras_seed(args: argparse.Namespace, config: ExperimentConfig) -> int:
    """Seed de onde saem as figuras — fixada de antemao, nunca a melhor no teste.

    Sem ``--figuras-seed``, usa a **primeira seed declarada** no config. Antes isto
    era um ``0`` escrito no codigo, que nao casaria com nada se o experimento
    declarasse ``seeds: [7, 11, 13]`` — e as figuras sumiriam em silencio.
    """
    return config.seeds[0] if args.figuras_seed is None else args.figuras_seed


def encontrar_execucoes(raiz: Path, nome: str) -> list[Path]:
    """Diretorios de execucao de uma versao, em ordem de seed."""
    return sorted(d for d in raiz.glob(f"{nome}_b*_s*") if (d / BEST_NAME).exists())


def carregar_execucao(run_dir: Path) -> tuple[ExperimentConfig, int]:
    """Configuracao e a seed **desta** execucao, a partir do manifesto.

    A seed vem de ``manifesto["run"]["seed"]``, nao do config: o config declara o
    conjunto de seeds do experimento (``seeds: [0, 42, 123]``), e cada diretorio
    corresponde a uma delas.
    """
    manifesto = read_manifest(run_dir)
    return ExperimentConfig.from_raw(manifesto["config"]), manifest_seed(manifesto)


def avaliar(
    run_dir: Path, *, device: str
) -> tuple[ExperimentConfig, int, EvaluationResult, int]:
    """Carrega o ``best.pt`` da execucao e o avalia no ``test_public``.

    Devolve ``(config, seed da execucao, resultado, n_parametros)``.
    """
    config, seed = carregar_execucao(run_dir)
    catalog = DataCatalog.open(config.data.root)
    num_classes = len(catalog.classes)

    # pretrained=False: os pesos vem do checkpoint, nao de download
    model = build_model(
        replace(config.model, pretrained=False), num_classes=num_classes
    ).to(device)
    load_checkpoint(run_dir / BEST_NAME, model=model, map_location=device)

    loader = labeled_loader(
        catalog,
        Split.TEST_PUBLIC,
        config.data,
        transform=build_augmentation("eval", image_size=config.data.image_size),
        shuffle=False,
    )
    resultado = evaluate(
        model, loader, split=Split.TEST_PUBLIC, num_classes=num_classes, device=device
    )
    return config, seed, resultado, count_parameters(model)


def tabela_markdown(
    resumos, parametros: dict[str, int], melhor_epoca: dict[str, list[int]]
) -> str:
    linhas = [
        "# Parte 0 — o piso",
        "",
        "F1 macro e acuracia no `test_public`, com o orcamento de 5 rotulos por classe.",
        "Media ± desvio amostral sobre as seeds. Epoca escolhida no `val`.",
        "",
        "| Versao | Parametros | Seeds | F1 macro | Acuracia | Melhor epoca (val) |",
        "|---|---:|---:|---|---|---|",
    ]
    for resumo in resumos:
        epocas = melhor_epoca.get(resumo.metodo, [])
        linhas.append(
            f"| `{resumo.metodo}` | {parametros.get(resumo.metodo, 0):,} | "
            f"{resumo.seeds} | {resumo.f1_formatado()} | "
            f"{resumo.acuracia_formatada()} | {', '.join(map(str, epocas)) or '—'} |"
        )
    linhas += [
        f"| teto de referencia (monitor, todos os rotulos) | — | — | {TETO_F1:.3f} | — | — |",
        "",
        "O teto vem do enunciado (ResNet-18 treinada com todos os rotulos do `train`).",
        "Nao foi reproduzido aqui; entra como linha de referencia na Parte 2.",
    ]
    return "\n".join(linhas) + "\n"


def escrever_f1_por_classe(
    caminho: Path, classes: Sequence[str], por_versao: dict[str, list[EvaluationResult]]
) -> Path:
    """F1 por classe, media sobre as seeds — o 'numero por classe' que o CLAUDE.md pede."""
    import csv
    import statistics

    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as handle:
        escritor = csv.writer(handle)
        escritor.writerow(["classe", "suporte", *(f"f1_{v}" for v in por_versao)])
        for indice, classe in enumerate(classes):
            algum = next(iter(por_versao.values()))[0]
            linha = [classe, algum.report.support[indice]]
            for resultados in por_versao.values():
                valores = [r.report.per_class_f1[indice] for r in resultados]
                linha.append(f"{statistics.fmean(valores):.4f}")
            escritor.writerow(linha)
    return caminho


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    SAIDA.mkdir(parents=True, exist_ok=True)

    linhas_orcamento: list[BudgetRow] = []
    por_versao: dict[str, list[EvaluationResult]] = {}
    parametros: dict[str, int] = {}
    melhor_epoca: dict[str, list[int]] = {}
    classes: Sequence[str] = ()

    for nome, metodo in VERSOES.items():
        execucoes = encontrar_execucoes(args.outputs, nome)
        if not execucoes:
            print(f"  (nenhuma execucao de {nome} em {args.outputs})")
            continue
        for run_dir in execucoes:
            config, seed, resultado, n_params = avaliar(run_dir, device=args.device)
            catalog = DataCatalog.open(config.data.root)
            classes = catalog.classes.labels

            linhas_orcamento.append(
                BudgetRow(
                    metodo=metodo,
                    orcamento=config.data.budget,
                    seed=seed,
                    f1_macro=resultado.macro_f1,
                    acuracia=resultado.accuracy,
                )
            )
            por_versao.setdefault(metodo, []).append(resultado)
            parametros[metodo] = n_params

            historico = json.loads(
                (run_dir / "history.json").read_text(encoding="utf-8")
            )
            if historico:
                melhor = max(historico, key=lambda linha: linha.get("val_macro_f1", -1))
                melhor_epoca.setdefault(metodo, []).append(int(melhor["epoch"]))

            print(
                f"  {run_dir.name:28} F1 {resultado.macro_f1:.4f} | "
                f"acc {resultado.accuracy:.4f}"
            )

            # figuras: seed fixada, nunca a melhor no teste
            if seed == figuras_seed(args, config) and nome == "p0_finetune":
                plot_confusion_matrix(
                    resultado.report.confusion,
                    classes,
                    SAIDA / "confusao_finetune.png",
                    title=f"Parte 0 (b) — test_public, seed {seed}",
                )
                plot_training_curve(
                    historico,
                    SAIDA / "curva_finetune.png",
                    title=f"Parte 0 (b) — ResNet-18 com 105 rotulos, seed {seed}",
                )

    if not linhas_orcamento:
        raise AppError(
            f"nenhuma execucao encontrada em {args.outputs}. "
            f"Rode scripts/train.py com configs/p0_*.yaml antes."
        )

    resumos = summarize(linhas_orcamento)
    (SAIDA / "tabela.md").write_text(
        tabela_markdown(resumos, parametros, melhor_epoca), encoding="utf-8"
    )
    escrever_f1_por_classe(SAIDA / "f1_por_classe.csv", classes, por_versao)
    update_budget_curve(CURVA_ORCAMENTO, linhas_orcamento)

    print()
    for resumo in resumos:
        print(
            f"  {resumo.metodo:16} F1 macro {resumo.f1_formatado():>16} | "
            f"acuracia {resumo.acuracia_formatada():>16}  ({resumo.seeds} seeds)"
        )
    print(f"\n  tabela:    {SAIDA / 'tabela.md'}")
    print(f"  por classe: {SAIDA / 'f1_por_classe.csv'}")
    print(f"  orcamento:  {CURVA_ORCAMENTO}")


if __name__ == "__main__":
    try:
        main()
    except AppError as erro:
        raise SystemExit(f"erro: {erro}")
