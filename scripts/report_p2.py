"""Relatorio da Parte 2 — a curva de orcamento.

Le ``curva_orcamento.csv``, que ja foi gerado pelas execucoes, e produz:

    outputs/p2/curva_orcamento.png   F1 macro contra n de rotulos, eixo x em log
    outputs/p2/tabela.md             o mesmo em numeros, com as seeds por ponto

Uso:
    python scripts/report_p2.py
    python scripts/report_p2.py --metodos piso_finetune fixmatch

Nao carrega modelo nem roda forward: os numeros do CSV vieram das execucoes, e
reprocessa-los daria o mesmo resultado por um custo de GPU. Por isso tambem **nao**
ha risco de vazamento aqui — o script nao tem acesso a nenhum conjunto de dados.

O que o CSV **nao** diz, e a tabela diz: quantos passos de otimizacao cada ponto
recebeu. Com epocas fixas, o orcamento de 1 rotulo treina em 21 imagens e o de 20 em
420, o que muda o numero de passos por epoca em 20x. E a mesma armadilha da D16, e a
coluna existe para que ninguem leia a curva sem ver isso.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import _bootstrap  # noqa: F401

from src.analysis.budget import (
    REFERENCE_CEILING_F1,
    BudgetSummary,
    read_budget_curve,
    summarize,
)
from src.analysis.figures import plot_budget_curve
from src.errors import AppError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAIDA = PROJECT_ROOT / "outputs" / "p2"
CURVA_ORCAMENTO = PROJECT_ROOT / "curva_orcamento.csv"

#: As duas curvas que o enunciado pede: o piso da Parte 0 (b) e o metodo da trilha.
#: A trilha do projeto e a **C**, e o tratamento de ruido escolhido no `val` foi o
#: filtro por confianca (ver docs/decisions.md, D20). O `fixmatch` da trilha B nao
#: entra por padrao: ela ficou como registro de aprendizado, nao como o metodo
#: entregue. As variantes (`piso_aug_weak`, `aluno_rotulos_suaves`, ...) entram com
#: `--metodos` quando interessarem.
METODOS_PADRAO = ("piso_finetune", "aluno_filtro_confianca")

#: Nome do arquivo com que uma trilha **publica** linhas horizontais para esta curva.
#: Contrato: qualquer `outputs/<qualquer>/referencias.json` com
#: ``{"rotulo": valor_de_f1_macro}`` e lido e desenhado como reta.
#:
#: Existe para resolver um conflito real. O enunciado manda incluir o professor na
#: curva da Trilha C, e o professor e zero-shot — nao usa rotulo nenhum, logo e uma
#: reta e nao uma curva. Mas este script e **neutro**: se ele importasse de
#: `src/tracks/c/`, apagar a trilha C o quebraria e
#: `tests/test_tracks_independence.py` falharia. Um arquivo de dados resolve: a
#: trilha publica o numero que ela mediu, este script le se existir, e apagar a
#: trilha faz a reta desaparecer sozinha. Tambem evita numero magico: o 0,350 do
#: professor e medido em `scripts/report_p1c.py`, nao digitado aqui.
ARQUIVO_DE_REFERENCIAS = "referencias.json"

#: Numero de classes do PA. Usado so para dizer quantas imagens tem cada orcamento.
NUM_CLASSES = 21

#: Tamanho do pool sem rotulo (as 6.505 imagens do `train`). Usado para explicar a
#: ressalva da leitura da curva, nao para calcular metrica nenhuma.
POOL_SIZE = 6505


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relatorio da Parte 2 do PA3")
    parser.add_argument("--curva", type=Path, default=CURVA_ORCAMENTO)
    parser.add_argument(
        "--metodos",
        nargs="+",
        default=list(METODOS_PADRAO),
        help=f"curvas a desenhar, na ordem (padrao: {' '.join(METODOS_PADRAO)})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="lote rotulado, para contar os passos por epoca (padrao: 16)",
    )
    parser.add_argument(
        "--sem-professor",
        action="store_true",
        help="omite as retas publicadas pelas trilhas (ex.: o professor da trilha C)",
    )
    return parser.parse_args()


def ler_referencias(raiz: Path) -> dict[str, float]:
    """Linhas horizontais publicadas pelas trilhas presentes.

    Varre ``<raiz>/*/referencias.json``. Nao importa trilha nenhuma: o acoplamento e
    por dado, nao por codigo. Arquivo ilegivel e ignorado com aviso, porque uma reta
    faltando nao justifica derrubar a figura obrigatoria da Parte 2.
    """
    referencias: dict[str, float] = {}
    for caminho in sorted(raiz.glob(f"*/{ARQUIVO_DE_REFERENCIAS}")):
        try:
            dados = json.loads(caminho.read_text(encoding="utf-8"))
            referencias.update({str(k): float(v) for k, v in dados.items()})
        except (OSError, ValueError, TypeError) as erro:
            print(f"  (ignorando {caminho}: {erro})")
    return referencias


def passos_por_epoca(orcamento: int, batch_size: int) -> int:
    """Passos de otimizacao por epoca quando a epoca e definida pelos **rotulos**.

    E o caso do piso: a epoca varre as ``21 * orcamento`` imagens rotuladas. O
    ``DataLoader`` nao descarta o ultimo lote incompleto, dai o teto.
    """
    return math.ceil(orcamento * NUM_CLASSES / batch_size)


def passagens_pelos_rotulos(orcamento: int, batch_size: int, mu: int) -> float:
    """Quantas vezes um metodo **guiado pelo pool** passa pelos rotulos, por epoca.

    E a contraparte da funcao acima para a Trilha C (e para o FixMatch). Quando a
    fonte primaria e o pool sem rotulo, a epoca tem ``POOL_SIZE / (batch_size * mu)``
    passos **independentemente do orcamento**, e o conjunto rotulado e ciclado dentro
    dela. Logo:

        passagens por epoca = passos_da_epoca * batch_size / (21 * orcamento)

    O numero cresce quando o orcamento **diminui** — o oposto da intuicao. Com 1
    rotulo por classe sao 21 imagens vistas ~44 vezes por epoca; com 20, as 420
    imagens sao vistas ~2 vezes. E a mesma armadilha da D16, e ela distorce a leitura
    da curva de orcamento se nao for dita.
    """
    passos = POOL_SIZE // (batch_size * mu)
    return passos * batch_size / (orcamento * NUM_CLASSES)


def tabela_markdown(
    resumos: Sequence[BudgetSummary],
    metodos: Sequence[str],
    batch_size: int,
    referencias: Mapping[str, float] | None = None,
) -> str:
    orcamentos = sorted({r.orcamento for r in resumos})
    por_chave = {(r.metodo, r.orcamento): r for r in resumos}

    linhas = [
        "# Parte 2 — A curva de orcamento",
        "",
        "F1 macro no `test_public` por orcamento de rotulos. Media ± desvio onde ha",
        "3 seeds; nos orcamentos de 1 e 20 o protocolo do enunciado pede uma seed, e",
        "o numero aparece sem desvio — nao e desvio zero, e desvio nao medido.",
        "",
        "| Orcamento | Imagens rotuladas | Passos/epoca (metodo guiado por rotulo) | "
        + " | ".join(f"`{m}`" for m in metodos)
        + " |",
        "|---:|---:|---:|" + "---|" * len(metodos),
    ]
    for orcamento in orcamentos:
        celulas = []
        for metodo in metodos:
            resumo = por_chave.get((metodo, orcamento))
            if resumo is None:
                celulas.append("—")
                continue
            sufixo = "" if resumo.seeds > 1 else " (1 seed)"
            celulas.append(f"{resumo.f1_formatado()}{sufixo}")
        linhas.append(
            f"| {orcamento} | {orcamento * NUM_CLASSES} | "
            f"{passos_por_epoca(orcamento, batch_size)} | " + " | ".join(celulas) + " |"
        )

    linhas += [
        f"| todos | 6.505 | — | teto de referencia: **{REFERENCE_CEILING_F1:.3f}** |"
        + " |" * (len(metodos) - 1),
        *(
            f"| nenhum | 0 | 0 | {nome}: **{valor:.3f}** |" + " |" * (len(metodos) - 1)
            for nome, valor in sorted((referencias or {}).items())
        ),
        "",
        "## A ressalva que a curva sozinha esconde",
        "",
        "A coluna de passos por epoca vale para um metodo cuja epoca e definida pelos",
        "**rotulos** — o piso. Com epocas fixas, o orcamento de 1 recebe 20x menos",
        "passos que o de 20, entao parte de qualquer subida da curva vem de mais",
        "rotulos e parte vem de mais atualizacoes de peso.",
        "",
        "Para o **aluno da Trilha C** a conta e outra, e vai na direcao contraria. A",
        "fonte primaria dele e o pool de 6.505 imagens, que nao encolhe com o",
        f"orcamento: a epoca tem sempre ~{POOL_SIZE // (16 * 7)} passos, e os rotulos sao",
        "ciclados dentro dela. Quantas vezes ele passa pelo conjunto rotulado:",
        "",
        "| Orcamento | Imagens rotuladas | Passagens pelos rotulos por epoca |",
        "|---:|---:|---:|",
        *(
            f"| {o} | {o * NUM_CLASSES} | ~{passagens_pelos_rotulos(o, 16, 7):.0f}x |"
            for o in (1, 5, 20)
        ),
        "",
        "Ou seja: **quanto menor o orcamento, mais o aluno memoriza cada rotulo.** Com",
        "1 rotulo por classe ele ve as mesmas 21 imagens ~44 vezes por epoca. Se o",
        "ponto de 1 rotulo sair ruim, essa e a primeira suspeita — e nao a falta de",
        "dado em si. Ver `docs/decisions.md`, D16, para o estrago que essa confusao",
        "fez na Parte 1.",
    ]
    return "\n".join(linhas) + "\n"


def main() -> None:
    args = parse_args()
    if not args.curva.exists():
        raise AppError(f"{args.curva} nao existe; rode os treinos antes")

    resumos = summarize(read_budget_curve(args.curva))
    disponiveis = {r.metodo for r in resumos}
    faltando = [m for m in args.metodos if m not in disponiveis]
    if faltando:
        raise AppError(
            f"sem linhas para {', '.join(faltando)} em {args.curva}; "
            f"ha: {', '.join(sorted(disponiveis))}"
        )

    SAIDA.mkdir(parents=True, exist_ok=True)
    referencias = (
        {} if args.sem_professor else ler_referencias(PROJECT_ROOT / "outputs")
    )
    series = {
        metodo: [
            (r.orcamento, r.f1_medio, r.f1_desvio, r.seeds)
            for r in resumos
            if r.metodo == metodo
        ]
        for metodo in args.metodos
    }
    figura = plot_budget_curve(
        series,
        SAIDA / "curva_orcamento.png",
        ceiling=REFERENCE_CEILING_F1,
        references=None if args.sem_professor else referencias,
        title="Parte 2 — F1 macro contra numero de rotulos",
    )
    tabela = SAIDA / "tabela.md"
    tabela.write_text(
        tabela_markdown(resumos, args.metodos, args.batch_size, referencias),
        encoding="utf-8",
    )

    for metodo in args.metodos:
        for orcamento, f1, desvio, seeds in sorted(series[metodo]):
            marca = f" ± {desvio:.3f}" if seeds > 1 else ""
            print(f"  {metodo:16} b={orcamento:<3} F1 {f1:.4f}{marca}  ({seeds} seeds)")
    print(f"\n  figura: {figura}")
    print(f"  tabela: {tabela}")


if __name__ == "__main__":
    main()
