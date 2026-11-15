"""Relatorio da Parte 1 — Trilha B.

Produz o que o enunciado exige e o que sustenta a apresentacao:

    outputs/p1/tabela.md                   piso, FixMatch e FixMatch+DA no test_public
    outputs/p1/ganho_por_classe.csv        ganho sobre o piso, classe a classe
    outputs/p1/pseudo_rotulos_tau_*.png    precisao e cobertura por classe vs tau
    outputs/p1/vies_classes.png            para onde vao os pseudo-rotulos
    outputs/p1/vies_classes.csv            os numeros por tras da figura
    curva_orcamento.csv                    linhas do metodo no orcamento de 5

Uso:
    python scripts/report_p1.py                  # comparacao final e diagnosticos
    python scripts/report_p1.py --sweep p1_tau   # escolhe tau no val, so isso

**Integridade experimental.** A escolha de tau sai de ``--sweep``, que compara F1
macro de **validacao** entre as execucoes via ``best_candidate``
(``src/evaluation/selection.py``) — funcao que levanta ``SplitPermissionError`` se
receber resultado de teste. As curvas de precisao e cobertura saem do
``test_public``, porque e o que o enunciado manda, e **nao alimentam escolha
nenhuma**: sao so diagnostico.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence

import _bootstrap  # noqa: F401
import numpy as np
from numpy.typing import NDArray

from src.analysis.budget import (
    REFERENCE_CEILING_F1,
    BudgetRow,
    summarize,
    update_budget_curve,
)

from src.tracks.b.figures import plot_label_distribution, plot_pseudo_label_curves
from src.tracks.b.pseudo_labels import (
    bias_ratio,
    label_distribution,
    pseudo_label_curve,
)
from src.data.splits import Split
from src.errors import AppError
from src.evaluation import EvaluationResult, best_candidate, evaluate
from src.evaluation.runs import eval_loader, load_run, probabilities
from src.models.checkpoint import BEST_NAME
from src.tracking import find_runs, resolve_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAIDA = PROJECT_ROOT / "outputs" / "p1"
CURVA_ORCAMENTO = PROJECT_ROOT / "curva_orcamento.csv"

#: Orcamento da Parte 1. O enunciado fixa o de 5 para a comparacao da trilha;
#: 1 e 20 existem so na curva da Parte 2 e nao entram nesta tabela.
ORCAMENTO_PADRAO = 5

#: Nome da execucao -> rotulo nas tabelas. A ordem e a da apresentacao: piso,
#: controle da aumentacao, metodo, metodo corrigido.
VERSOES = {
    "p0_finetune": "piso_finetune",
    "p0_finetune_weak": "piso_aug_weak",
    "p1_fixmatch": "fixmatch",
    "p1_fixmatch_da": "fixmatch_da",
}

#: Variantes de onde saem os diagnosticos de pseudo-rotulo.
DIAGNOSTICAR = ("p0_finetune", "p1_fixmatch", "p1_fixmatch_da")


def pasta_de_saida(orcamento: int) -> Path:
    """Onde os artefatos deste orcamento sao gravados.

    O orcamento padrao escreve direto em ``outputs/p1``, que e o que o README e a
    apresentacao citam. Os outros ganham subpasta ``b<orcamento>``, para que uma
    chamada da Parte 2 nao sobrescreva a tabela do orcamento de 5 — as duas
    afirmam coisas diferentes e as duas precisam continuar existindo.
    """
    if orcamento == ORCAMENTO_PADRAO:
        return SAIDA
    return SAIDA / f"b{orcamento}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relatorio da Parte 1 do PA3")
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--sweep",
        default=None,
        metavar="PREFIXO",
        help="escolhe tau comparando no val as execucoes que comecam com PREFIXO",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.80,
        help="limiar usado nas figuras de diagnostico (padrao: 0.80)",
    )
    parser.add_argument(
        "--orcamento",
        type=int,
        default=ORCAMENTO_PADRAO,
        help=f"orcamento de rotulos a reportar (padrao: {ORCAMENTO_PADRAO})",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


# ------------------------------------------------------------------ execucoes
# ------------------------------------------------------------------- varredura
def escolher_tau(raiz: Path, prefixo: str, *, device: str) -> None:
    """Compara no ``val`` as execucoes da varredura e imprime a vencedora."""
    execucoes = sorted(
        d for d in raiz.glob(f"{prefixo}*_b*_s*") if (d / BEST_NAME).exists()
    )
    if not execucoes:
        raise AppError(
            f"nenhuma execucao comecando com {prefixo!r} em {raiz}. "
            f"Rode a varredura antes (ver `make parte1-sweep`)."
        )

    candidatos: dict[str, EvaluationResult] = {}
    for run_dir in execucoes:
        execucao = load_run(run_dir, device=device)
        resultado = evaluate(
            execucao.model,
            eval_loader(execucao, Split.VAL),
            split=Split.VAL,
            num_classes=execucao.num_classes,
            device=device,
        )
        tau = execucao.config.strategy.params.get("tau", "?")
        rotulo = f"{run_dir.name}  tau={tau}"
        candidatos[rotulo] = resultado
        print(f"  {rotulo:44} val F1 {resultado.macro_f1:.4f}")

    # best_candidate recusa resultado que nao venha do val
    vencedor = best_candidate(candidatos)
    print(f"\n  escolhido no val: {vencedor}")
    print("  (o test_public nao participou desta escolha)")


# -------------------------------------------------------------------- tabelas
def tabela_markdown(resumos, por_versao: Mapping[str, list[EvaluationResult]]) -> str:
    piso = next((r for r in resumos if r.metodo == "piso_finetune"), None)
    linhas = [
        "# Parte 1 — Trilha B: FixMatch",
        "",
        "F1 macro e acuracia no `test_public`, orcamento de 5 rotulos por classe.",
        "Media ± desvio amostral sobre as seeds. Epoca escolhida no `val`.",
        "",
        "| Metodo | Seeds | F1 macro | Acuracia | Ganho sobre o piso |",
        "|---|---:|---|---|---|",
    ]
    for resumo in resumos:
        ganho = (
            f"{resumo.f1_medio - piso.f1_medio:+.3f}"
            if piso is not None and resumo.metodo != "piso_finetune"
            else "—"
        )
        linhas.append(
            f"| `{resumo.metodo}` | {resumo.seeds} | {resumo.f1_formatado()} | "
            f"{resumo.acuracia_formatada()} | {ganho} |"
        )
    linhas += [
        f"| teto de referencia (monitor, todos os rotulos) | — | {REFERENCE_CEILING_F1:.3f} | — | — |",
        "",
        "`piso_aug_weak` e o **controle do confundidor**: o piso treinado com a mesma",
        "aumentacao fraca do FixMatch. O que estiver acima dele e ganho dos dados sem",
        "rotulo; o que estiver entre ele e `piso_finetune` veio da aumentacao.",
    ]
    return "\n".join(linhas) + "\n"


def escrever_ganho_por_classe(
    caminho: Path,
    classes: Sequence[str],
    por_versao: Mapping[str, list[EvaluationResult]],
) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    versoes = list(por_versao)
    with caminho.open("w", newline="", encoding="utf-8") as handle:
        escritor = csv.writer(handle)
        escritor.writerow(["classe", "suporte", *(f"f1_{v}" for v in versoes)])
        referencia = next(iter(por_versao.values()))[0]
        for indice, classe in enumerate(classes):
            linha = [classe, referencia.report.support[indice]]
            for versao in versoes:
                valores = [r.report.per_class_f1[indice] for r in por_versao[versao]]
                linha.append(f"{statistics.fmean(valores):.4f}")
            escritor.writerow(linha)
    return caminho


def escrever_vies_por_classe(
    caminho: Path,
    classes: Sequence[str],
    distribuicoes: Mapping[str, NDArray[np.int64]],
    contagem_real: NDArray[np.int64],
) -> Path:
    """Os numeros por tras de ``vies_classes.png``.

    A figura mostra a forma; o CSV da o valor que vai para a apresentacao. A razao
    de vies resume cada classe num numero: ``1`` e representacao fiel, acima de 1 a
    classe recebe mais pseudo-rotulos do que o tamanho dela justifica, abaixo de 1
    recebe menos. Zero significa classe sem nenhum pseudo-rotulo — e a que o
    diagnostico obrigatorio 2 procura.
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    metodos = list(distribuicoes)
    razoes = {m: bias_ratio(distribuicoes[m], contagem_real) for m in metodos}
    with caminho.open("w", newline="", encoding="utf-8") as handle:
        escritor = csv.writer(handle)
        escritor.writerow(
            [
                "classe",
                "suporte",
                *(f"pseudo_{m}" for m in metodos),
                *(f"razao_vies_{m}" for m in metodos),
            ]
        )
        for indice, classe in enumerate(classes):
            escritor.writerow(
                [
                    classe,
                    int(contagem_real[indice]),
                    *(int(distribuicoes[m][indice]) for m in metodos),
                    *(f"{razoes[m][indice]:.3f}" for m in metodos),
                ]
            )
    return caminho


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)

    if args.sweep:
        escolher_tau(args.outputs, args.sweep, device=args.device)
        return

    saida = pasta_de_saida(args.orcamento)
    saida.mkdir(parents=True, exist_ok=True)
    linhas: list[BudgetRow] = []
    por_versao: dict[str, list[EvaluationResult]] = {}
    distribuicoes: dict[str, np.ndarray] = {}
    classes: Sequence[str] = ()
    verdadeiros_ref: np.ndarray | None = None

    for nome, metodo in VERSOES.items():
        execucoes = find_runs(
            args.outputs, nome, budget=args.orcamento, requires=BEST_NAME
        )
        if not execucoes:
            print(f"  (sem execucoes de {nome})")
            continue
        for run_dir in execucoes:
            execucao = load_run(run_dir, device=args.device)
            config, seed = execucao.config, execucao.seed
            model, catalog = execucao.model, execucao.catalog
            classes = catalog.classes.labels
            probs, verdadeiros = probabilities(
                model, eval_loader(execucao, Split.TEST_PUBLIC), device=args.device
            )
            verdadeiros_ref = verdadeiros
            resultado = evaluate(
                model,
                eval_loader(execucao, Split.TEST_PUBLIC),
                split=Split.TEST_PUBLIC,
                num_classes=len(classes),
                device=args.device,
            )
            linhas.append(
                BudgetRow(
                    metodo=metodo,
                    orcamento=config.data.budget,
                    seed=seed,
                    f1_macro=resultado.macro_f1,
                    acuracia=resultado.accuracy,
                )
            )
            por_versao.setdefault(metodo, []).append(resultado)
            print(
                f"  {run_dir.name:30} F1 {resultado.macro_f1:.4f} | "
                f"acc {resultado.accuracy:.4f}"
            )

            # diagnosticos saem da primeira seed declarada, fixada de antemao
            if nome in DIAGNOSTICAR and seed == config.seeds[0]:
                curva = pseudo_label_curve(probs, verdadeiros, num_classes=len(classes))
                plot_pseudo_label_curves(
                    curva,
                    classes,
                    saida / f"pseudo_rotulos_tau_{metodo}.png",
                    title=f"Pseudo-rotulador: {metodo} — test_public, seed {seed}",
                )
                distribuicoes[metodo] = label_distribution(
                    probs, tau=args.tau, num_classes=len(classes)
                )
                (saida / f"curva_tau_{metodo}.json").write_text(
                    json.dumps(
                        {
                            "taus": list(curva.taus),
                            "mask_rate": curva.mask_rate.tolist(),
                            "overall_precision": curva.overall_precision.tolist(),
                            "alive_classes": curva.alive_classes.tolist(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    if not linhas:
        raise AppError(f"nenhuma execucao encontrada em {args.outputs}")

    resumos = summarize(linhas)
    (saida / "tabela.md").write_text(
        tabela_markdown(resumos, por_versao), encoding="utf-8"
    )
    escrever_ganho_por_classe(saida / "ganho_por_classe.csv", classes, por_versao)
    if distribuicoes and verdadeiros_ref is not None:
        contagem_real = np.bincount(verdadeiros_ref, minlength=len(classes))
        plot_label_distribution(
            distribuicoes,
            contagem_real,
            classes,
            saida / "vies_classes.png",
            title=f"Para onde vao os pseudo-rotulos (tau={args.tau})",
        )
        escrever_vies_por_classe(
            saida / "vies_classes.csv", classes, distribuicoes, contagem_real
        )
    update_budget_curve(CURVA_ORCAMENTO, linhas)

    print()
    for resumo in resumos:
        print(
            f"  {resumo.metodo:18} F1 macro {resumo.f1_formatado():>16} | "
            f"acuracia {resumo.acuracia_formatada():>16}  ({resumo.seeds} seeds)"
        )
    print(f"\n  tabela:     {saida / 'tabela.md'}")
    print(f"  por classe: {saida / 'ganho_por_classe.csv'}")
    print(f"  orcamento:  {CURVA_ORCAMENTO}")


if __name__ == "__main__":
    try:
        main()
    except AppError as erro:
        raise SystemExit(f"erro: {erro}")
