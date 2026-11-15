"""Relatorio da Parte 1 — Trilha C (professor e aluno).

    python scripts/report_p1c.py
    python scripts/report_p1c.py --device cpu

Produz em ``outputs/p1c/``:

    tabela.md                 professores, alunos e o piso no test_public
    ruido_<professor>.png     a matriz de ruido — obrigatorio 2
    ruido_<professor>.csv     as confusoes sistematicas em numeros
    calibracao.png            confianca do professor quando acerta vs quando erra
    aluno_vs_professor.png    obrigatorio 3: em quais classes o aluno supera
    aluno_vs_professor.csv    o mesmo em numeros, com a pureza do pseudo-rotulo
    noite_vs_dia.md           a hipotese "noite" do obrigatorio 2, com o confundidor
    referencias.json          o F1 do professor, para a curva da Parte 2 desenhar
    curva_orcamento.csv       linhas dos alunos no orcamento de 5

**Integridade.** Este script so reporta: nao chama nenhuma funcao de
``src.evaluation.selection``, e a epoca de cada aluno foi escolhida no ``val``
durante o treino. O professor e medido no ``test_public`` porque o enunciado manda
("para voces medirem o professor"); ele nunca viu o ``val``.
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
from PIL import Image

from src.analysis.budget import (
    REFERENCE_CEILING_F1,
    BudgetRow,
    summarize,
    update_budget_curve,
)
from src.analysis.figures import plot_confusion_matrix
from src.analysis.images import NIGHT_SATURATION_THRESHOLD, mean_saturation
from src.data.catalog import DataCatalog
from src.data.splits import Split
from src.errors import AppError
from src.evaluation import EvaluationResult, evaluate
from src.evaluation.runs import best_validation_metric, eval_loader, load_run
from src.metrics import classification_report
from src.models.checkpoint import BEST_NAME
from src.tracks.c.analysis import (
    compare_groups,
    confidence_by_correctness,
    noise_matrix,
    student_vs_teacher,
)
from src.tracks.c.figures import plot_student_vs_teacher, plot_teacher_calibration
from src.tracks.c.teacher import TeacherCache, available_teachers, cache_path
from src.tracking import find_runs, resolve_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAIDA = PROJECT_ROOT / "outputs" / "p1c"
CACHES = PROJECT_ROOT / "outputs" / "teacher"
CURVA_ORCAMENTO = PROJECT_ROOT / "curva_orcamento.csv"

#: Execucao -> rotulo nas tabelas, na ordem da apresentacao: sem tratamento,
#: a escolha ingenua, a escolha que a trilha B recomenda.
ALUNOS = {
    "p1c_hard": "aluno_sem_tratamento",
    "p1c_filter": "aluno_filtro_confianca",
    "p1c_soft": "aluno_rotulos_suaves",
}

#: Piso da Parte 0, para a tabela. E a linha que importa: um aluno que nao passa o
#: piso nao justificou o professor.
PISO = {"p0_finetune": "piso_finetune", "p0_finetune_weak": "piso_aug_weak"}

ORCAMENTO_PADRAO = 5

#: Formato das diferencas controladas na tabela de noite vs dia.
PORCENTAGEM = "{:+.1%}"

#: Limiar do modo `filter`, so para anotar a figura de calibracao.
TAU_TESTADO = 0.80


def pasta_de_saida(orcamento: int) -> Path:
    """Onde os artefatos deste orcamento sao gravados.

    O orcamento padrao escreve direto em ``outputs/p1c``, que e o que o README e a
    apresentacao citam. Os outros ganham subpasta ``b<orcamento>``, para que as
    chamadas da Parte 2 (``--orcamento 1`` e ``--orcamento 20``) nao sobrescrevam a
    tabela do orcamento de 5 — as tres afirmam coisas diferentes.

    O ``referencias.json`` e a excecao e fica sempre na raiz: o professor e zero-shot,
    logo o F1 dele nao depende de orcamento nenhum.
    """
    if orcamento == ORCAMENTO_PADRAO:
        return SAIDA
    return SAIDA / f"b{orcamento}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relatorio da Trilha C do PA3")
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--caches", type=Path, default=CACHES)
    parser.add_argument("--orcamento", type=int, default=ORCAMENTO_PADRAO)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


# ============================================================ o professor
def medir_professor(
    nome: str, caches: Path, catalog: DataCatalog
) -> tuple[object, np.ndarray, np.ndarray, list[str]]:
    """Relatorio, probabilidades e verdade do professor no ``test_public``.

    Casa por ``image_id``, nunca por posicao: o cache e o CSV de rotulos nao
    precisam estar na mesma ordem, e confiar na ordem produz numero plausivel e
    errado.
    """
    cache = TeacherCache.load(cache_path(caches, nome, Split.TEST_PUBLIC))
    verdade = {ex.image_id: ex.label_index for ex in catalog.labeled(Split.TEST_PUBLIC)}
    linha = cache.index()
    ids = [i for i in cache.image_ids if i in verdade]
    if not ids:
        raise AppError(f"nenhum id do cache de {nome} casa com os rotulos")
    y = np.array([verdade[i] for i in ids], dtype=np.int64)
    probs = cache.probabilities[[linha[i] for i in ids]]
    relatorio = classification_report(
        y, probs.argmax(axis=1), num_classes=len(catalog.classes)
    )
    return relatorio, probs, y, ids


def escrever_confusoes(caminho: Path, matriz) -> Path:
    """As confusoes sistematicas do professor — o texto do obrigatorio 2."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as handle:
        escritor = csv.writer(handle)
        escritor.writerow(["classe_verdadeira", "o_professor_disse", "taxa", "imagens"])
        for verdadeira, dita, taxa, n in matriz.worst_confusions(25):
            escritor.writerow([verdadeira, dita, f"{taxa:.4f}", n])
    return caminho


def analise_noite(
    probs: np.ndarray,
    y: np.ndarray,
    ids: Sequence[str],
    catalog: DataCatalog,
    classes: Sequence[str],
):
    """A hipotese "noite" do obrigatorio 2, com o confundidor controlado.

    Camera trap noturna usa infravermelho, e a imagem sai quase sem saturacao —
    proxy calculado da propria imagem, sem rotulo novo nem metadado externo.
    """
    saturacao = np.array(
        [
            mean_saturation(Image.open(catalog.image_path(Split.TEST_PUBLIC, i)))
            for i in ids
        ]
    )
    return compare_groups(
        probs,
        y,
        saturacao < NIGHT_SATURATION_THRESHOLD,
        class_names=classes,
        group_names=("noite (IR)", "dia (cor)"),
    )


def escrever_noite(caminho: Path, comparacoes: Mapping[str, object]) -> Path:
    linhas = [
        "# Trilha C — onde o professor erra: a hipotese da noite",
        "",
        "O obrigatorio 2 pede dizer onde o professor erra de forma sistematica e",
        "nomeia tres suspeitas: **noite**, **animais pequenos** e **classes",
        "parecidas**. Duas sao mensuraveis com o que o pacote da:",
        "",
        "* **classes parecidas** — sai da matriz de ruido (`ruido_*.csv`);",
        "* **noite** — camera trap noturna usa infravermelho, e a imagem sai quase",
        "  sem saturacao. Tratamos como noturna a imagem com saturacao media abaixo",
        f"  de {NIGHT_SATURATION_THRESHOLD:.0f} (escala 0-255), medida da propria imagem.",
        "",
        "**Animais pequenos nao e mensuravel aqui.** Precisaria da caixa do animal, e",
        "o pacote nao traz anotacao de deteccao. Afirmar que o professor erra em",
        "animais pequenos sem medir seria inventar. O que da para afirmar e que o",
        "pre-processamento do CLIP faz `CenterCrop`, logo animal na borda e descartado",
        "antes do modelo ver — contra a nossa D9, que redimensiona o quadro inteiro.",
        "",
        "## O confundidor, que inverte a resposta",
        "",
        "| Professor | Acuracia a noite | Acuracia de dia | Bruto | **Controlando a classe** | Classes comparaveis |",
        "|---|---|---|---|---|---|",
    ]
    for nome, c in sorted(comparacoes.items()):
        marca = " ⚠" if c.confounded() else ""
        linhas.append(
            f"| `{nome}` | {c.raw_means[0]:.1%} (n={c.raw_counts[0]}) | "
            f"{c.raw_means[1]:.1%} (n={c.raw_counts[1]}) | "
            f"{c.raw_delta():+.1%} | **{c.controlled_text(PORCENTAGEM)}**{marca} | "
            f"{c.comparable_classes} de {c.total_classes} |"
        )
    linhas += [
        "",
        "A coluna bruta compara **populacoes diferentes**: a noite e dominada por",
        "`virginia_opossum` e `american_black_bear`, e urso e a melhor classe do",
        "professor. Isso infla a acuracia noturna sem que o professor tenha ficado",
        "melhor no escuro. A coluna controlada compara so dentro de cada classe com",
        "suporte nos dois regimes, e e ela que responde a pergunta.",
        "",
        "O ⚠ marca os casos em que as duas colunas apontam para lados opostos — ou",
        "seja, em que a leitura ingenua estaria errada.",
        "",
        "**Ressalva.** Poucas classes tem suporte nos dois regimes, e algumas celulas",
        "tem 5 imagens. Uma celula dessas move a media ponderada varios pontos, entao",
        "o numero controlado indica direcao, nao magnitude confiavel.",
        "",
        "## Por classe",
        "",
    ]
    for nome, c in sorted(comparacoes.items()):
        linhas += [
            f"### `{nome}`",
            "",
            "| Classe | n noite | Acuracia noite | n dia | Acuracia dia | Delta |",
            "|---|---:|---|---:|---|---|",
        ]
        for classe, n_noite, ac_noite, n_dia, ac_dia in c.per_class:
            linhas.append(
                f"| {classe} | {n_noite} | {ac_noite:.1%} | {n_dia} | "
                f"{ac_dia:.1%} | {ac_noite - ac_dia:+.1%} |"
            )
        linhas.append("")
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return caminho


# ============================================================ os alunos
def avaliar_alunos(
    raiz: Path, orcamento: int, *, device: str
) -> tuple[list[BudgetRow], dict[str, list[EvaluationResult]], tuple[str, ...]]:
    linhas: list[BudgetRow] = []
    por_versao: dict[str, list[EvaluationResult]] = {}
    classes: tuple[str, ...] = ()
    for nome, rotulo in {**ALUNOS, **PISO}.items():
        execucoes = find_runs(raiz, nome, budget=orcamento, requires=BEST_NAME)
        if not execucoes:
            print(f"  (sem execucoes de {nome})")
            continue
        for run_dir in execucoes:
            execucao = load_run(run_dir, device=device)
            classes = execucao.classes
            resultado = evaluate(
                execucao.model,
                eval_loader(execucao, Split.TEST_PUBLIC),
                split=Split.TEST_PUBLIC,
                num_classes=execucao.num_classes,
                device=device,
            )
            linhas.append(
                BudgetRow(
                    metodo=rotulo,
                    orcamento=execucao.config.data.budget,
                    seed=execucao.seed,
                    f1_macro=resultado.macro_f1,
                    acuracia=resultado.accuracy,
                )
            )
            por_versao.setdefault(rotulo, []).append(resultado)
            print(
                f"  {run_dir.name:28} F1 {resultado.macro_f1:.4f} | "
                f"acc {resultado.accuracy:.4f}"
            )
    return linhas, por_versao, classes


def media_por_classe(resultados: Sequence[EvaluationResult]) -> np.ndarray:
    return np.array(
        [
            statistics.fmean(r.report.per_class_f1[i] for r in resultados)
            for i in range(resultados[0].report.num_classes)
        ]
    )


def tabela_markdown(resumos, professores: Mapping[str, object]) -> str:
    linhas = [
        "# Parte 1 — Trilha C: professor e aluno",
        "",
        "F1 macro e acuracia no `test_public`, orcamento de 5 rotulos por classe.",
        "A epoca de cada aluno foi escolhida no `val`; o professor e zero-shot e nao",
        "treina nem escolhe nada.",
        "",
        "| Metodo | Seeds | F1 macro | Acuracia |",
        "|---|---:|---|---|",
    ]
    for nome, (relatorio, descricao) in sorted(professores.items()):
        linhas.append(
            f"| professor `{nome}` ({descricao}) | — | {relatorio.macro_f1:.3f} | "
            f"{relatorio.accuracy:.3f} |"
        )
    for resumo in resumos:
        linhas.append(
            f"| `{resumo.metodo}` | {resumo.seeds} | {resumo.f1_formatado()} | "
            f"{resumo.acuracia_formatada()} |"
        )
    linhas += [
        f"| teto de referencia (todos os rotulos) | — | {REFERENCE_CEILING_F1:.3f} | — |",
        "",
        "Os professores sao **zero-shot**: nenhum dos 105 rotulos, nenhuma epoca, e a",
        "cabeca de classificacao construida dos nomes das classes via codificador de",
        "texto. Os dois sao ViT-B/16 — a diferenca entre eles mede o pre-treino.",
    ]
    return "\n".join(linhas) + "\n"


def escrever_aluno_vs_professor(caminho: Path, comparacao) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    linhas = comparacao.as_rows()
    with caminho.open("w", newline="", encoding="utf-8") as handle:
        escritor = csv.DictWriter(handle, fieldnames=list(linhas[0]))
        escritor.writeheader()
        for linha in linhas:
            escritor.writerows([linha])
    return caminho


def aluno_escolhido_no_val(raiz: Path, orcamento: int, resumos):
    """Qual tratamento de ruido reportar, decidido no ``val``.

    Devolve o ``BudgetSummary`` do aluno com melhor F1 macro de **validacao**, ou
    ``None`` se nenhum aluno tiver execucao. A comparacao nao carrega modelo e nao
    toca em dado: le o ``val_macro_f1`` que o treino gravou por epoca.
    """
    candidatos: list[tuple[float, int, object]] = []
    for nome, rotulo in ALUNOS.items():
        resumo = next((r for r in resumos if r.metodo == rotulo), None)
        if resumo is None:
            continue
        for run_dir in find_runs(raiz, nome, budget=orcamento, requires=BEST_NAME):
            f1, epoca = best_validation_metric(run_dir)
            candidatos.append((f1, epoca, resumo))
            print(f"  {run_dir.name:28} val F1 {f1:.4f} (epoca {epoca})")
    if not candidatos:
        return None
    melhor_f1, _epoca, melhor = max(candidatos, key=lambda c: c[0])
    print(f"  escolhido no val: {melhor.metodo} (val F1 {melhor_f1:.4f})")
    print("  (o test_public nao participou desta escolha)")
    return melhor


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    catalog = DataCatalog.open(PROJECT_ROOT / "data")
    classes = catalog.classes.labels
    saida = pasta_de_saida(args.orcamento)
    saida.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- o professor
    professores: dict[str, object] = {}
    calibracoes: dict[str, tuple[float, float]] = {}
    noites: dict[str, object] = {}
    matrizes: dict[str, object] = {}
    probabilidades_professor: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    print("##### professores (test_public — o enunciado libera para medi-los)")
    for nome in available_teachers():
        caminho = cache_path(args.caches, nome, Split.TEST_PUBLIC)
        if not caminho.exists():
            print(f"  (sem cache de {nome}; rode scripts/teacher.py)")
            continue
        relatorio, probs, y, ids = medir_professor(nome, args.caches, catalog)
        professores[nome] = (relatorio, "ViT-B/16 zero-shot")
        probabilidades_professor[nome] = (probs, y)
        matriz = noise_matrix(y, probs.argmax(axis=1), class_names=classes)
        matrizes[nome] = matriz
        calibracoes[nome] = confidence_by_correctness(probs, y)
        noites[nome] = analise_noite(probs, y, ids, catalog, classes)

        plot_confusion_matrix(
            matriz.counts,
            classes,
            saida / f"ruido_{nome}.png",
            title=f"Matriz de ruido dos pseudo-rotulos — professor {nome}",
        )
        escrever_confusoes(saida / f"ruido_{nome}.csv", matriz)
        mudas = matriz.collapsed_classes()
        acerto, erro = calibracoes[nome]
        print(
            f"  {nome:9} F1 {relatorio.macro_f1:.4f} | acc {relatorio.accuracy:.4f} | "
            f"confianca acerto {acerto:.3f} vs erro {erro:.3f} "
            f"(separacao {acerto - erro:+.3f})"
        )
        if mudas:
            print(f"            classes que ele nunca emite: {', '.join(mudas)}")

    if not professores:
        raise AppError(
            f"nenhum cache de professor em {args.caches}. "
            f"Rode `python scripts/teacher.py` primeiro."
        )

    plot_teacher_calibration(calibracoes, saida / "calibracao.png", tau=TAU_TESTADO)
    escrever_noite(saida / "noite_vs_dia.md", noites)

    # publica o F1 dos professores para a curva da Parte 2. O professor e zero-shot,
    # logo nao depende do orcamento: na curva ele e uma reta, e o enunciado manda
    # inclui-lo. Vai por **arquivo** e nao por import porque `scripts/report_p2.py`
    # e neutro — ele nao pode depender desta trilha (ver
    # tests/test_tracks_independence.py) — e porque assim o numero e o que medimos,
    # nao um literal digitado em dois lugares.
    (SAIDA / "referencias.json").write_text(
        json.dumps(
            {
                f"professor {nome} (zero-shot)": relatorio.macro_f1
                for nome, (relatorio, _descricao) in sorted(professores.items())
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------- os alunos
    print("\n##### alunos")
    linhas, por_versao, classes_run = avaliar_alunos(
        args.outputs, args.orcamento, device=device
    )
    if not linhas:
        print("  (nenhum aluno treinado ainda)")
    else:
        resumos = summarize(linhas)
        (saida / "tabela.md").write_text(
            tabela_markdown(resumos, professores), encoding="utf-8"
        )
        update_budget_curve(CURVA_ORCAMENTO, linhas)

        # obrigatorio 3: o aluno contra o professor, classe a classe.
        #
        # Qual aluno entra na figura sai do **val**, nao do test_public. Escolher
        # pelo F1 de teste — ainda que so para decidir o que desenhar — seria usar o
        # conjunto de reporte para tomar decisao, e o enunciado proibe. A metrica de
        # validacao vem do history.json, que por construcao nao contem metrica de
        # teste nenhuma (o laco de treino nunca ve o test_public).
        melhor = aluno_escolhido_no_val(args.outputs, args.orcamento, resumos)
        professor_do_aluno = "clip"  # declarado nos configs dos alunos
        if melhor is not None and professor_do_aluno in matrizes:
            relatorio_professor = professores[professor_do_aluno][0]
            comparacao = student_vs_teacher(
                class_names=classes_run or classes,
                support=relatorio_professor.support,
                teacher_f1=relatorio_professor.per_class_f1,
                student_f1=media_por_classe(por_versao[melhor.metodo]),
                noise=matrizes[professor_do_aluno],
            )
            plot_student_vs_teacher(
                comparacao,
                saida / "aluno_vs_professor.png",
                title=(
                    f"{melhor.metodo} contra o professor {professor_do_aluno} "
                    f"(test_public)"
                ),
            )
            escrever_aluno_vs_professor(saida / "aluno_vs_professor.csv", comparacao)
            ganhou = comparacao.student_wins()
            print(
                f"\n  melhor aluno no test_public: {melhor.metodo} "
                f"({melhor.f1_formatado()})"
            )
            print(
                f"  supera o professor em {len(ganhou)} de "
                f"{len(comparacao.class_names)} classes"
            )
            for nome, ganho in ganhou[:5]:
                print(f"    {nome:24} {ganho:+.3f}")

        print()
        for resumo in resumos:
            print(
                f"  {resumo.metodo:24} F1 macro {resumo.f1_formatado():>16} | "
                f"acuracia {resumo.acuracia_formatada():>16}  ({resumo.seeds} seeds)"
            )

    print(f"\n  artefatos: {saida}")


if __name__ == "__main__":
    try:
        main()
    except AppError as erro:
        raise SystemExit(f"erro: {erro}")
