"""Figuras da trilha C.

A **matriz de ruido** nao esta aqui: ela e uma matriz de confusao normalizada por
linha, e ``src.analysis.figures.plot_confusion_matrix`` ja desenha exatamente isso.
Duplicar um heatmap para chama-lo de outro nome seria codigo morto com outro rotulo.
O que esta aqui e o que nao existia:

``plot_student_vs_teacher``  o aluno contra o professor, classe a classe
``plot_teacher_calibration`` a confianca do professor quando acerta e quando erra

Reusam ``prepare_path`` e ``DPI`` de ``src.analysis.figures``, que e infraestrutura
comum. Nao importam nada de outra trilha.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem display: roda no Colab, em CI e por ssh
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.analysis.figures import DPI, prepare_path  # noqa: E402


def plot_student_vs_teacher(
    comparison,
    path: Path | str,
    *,
    title: str = "O aluno supera o professor em quais classes?",
) -> Path:
    """Responde o obrigatorio 3 com uma figura, classe a classe.

    Duas leituras na mesma imagem. A esquerda, as barras do professor e do aluno,
    ordenadas pelo ganho do aluno — quem esta no topo e onde a destilacao funcionou.
    A direita, o ganho contra a **pureza do pseudo-rotulo** daquela classe: se a
    nuvem descer da esquerda para a direita, a leitura e "o aluno ganha onde o rotulo
    do professor e sujo", que e a explicacao do "por que" que o enunciado pede.
    """
    path = prepare_path(path)
    ganho = comparison.gain()
    ordem = np.argsort(ganho)  # pior embaixo, melhor no topo
    nomes = [comparison.class_names[i] for i in ordem]
    posicoes = np.arange(len(nomes))

    figura, (esquerda, direita) = plt.subplots(
        1, 2, figsize=(14, 8.2), width_ratios=[1.6, 1]
    )

    esquerda.barh(
        posicoes + 0.2,
        comparison.teacher_f1[ordem],
        height=0.4,
        color="tab:gray",
        label="professor (zero-shot)",
    )
    esquerda.barh(
        posicoes - 0.2,
        comparison.student_f1[ordem],
        height=0.4,
        color="tab:blue",
        label="aluno (destilado)",
    )
    esquerda.set_yticks(posicoes)
    esquerda.set_yticklabels(
        [f"{n}  (n={int(comparison.support[i])})" for n, i in zip(nomes, ordem)],
        fontsize=8,
    )
    esquerda.set_xlabel("F1 da classe no test_public")
    esquerda.set_xlim(0, 1)
    esquerda.grid(axis="x", alpha=0.25)
    esquerda.legend(fontsize=8, loc="lower right")
    esquerda.set_title("por classe, ordenado pelo ganho do aluno")

    cores = ["tab:green" if g > 0 else "tab:red" for g in ganho]
    direita.scatter(comparison.label_purity, ganho, c=cores, s=45, zorder=3)
    for i, nome in enumerate(comparison.class_names):
        if abs(ganho[i]) > 0.15:
            direita.annotate(
                nome,
                (comparison.label_purity[i], ganho[i]),
                fontsize=6.5,
                xytext=(4, 2),
                textcoords="offset points",
            )
    direita.axhline(0, color="black", linewidth=1)
    direita.set_xlabel("pureza do pseudo-rotulo do professor naquela classe")
    direita.set_ylabel("F1 do aluno - F1 do professor")
    direita.set_xlim(-0.03, 1.03)
    direita.grid(alpha=0.25)
    direita.set_title("o aluno ganha onde o rotulo e sujo?")

    figura.suptitle(title)
    figura.tight_layout()
    figura.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figura)
    return path


def plot_teacher_calibration(
    series: dict[str, tuple[float, float]],
    path: Path | str,
    *,
    tau: float | None = None,
    title: str = "O professor e calibrado? (confianca quando acerta vs quando erra)",
) -> Path:
    """A figura que diz se filtrar por confianca pode funcionar.

    Se as duas barras de um professor tiverem altura parecida, a confianca dele nao
    separa acerto de erro, e nenhum limiar vai limpar o pseudo-rotulo — so vai
    descartar imagens. E a previsao que a trilha B deixou (D17), aqui em forma
    testavel antes de treinar qualquer aluno.
    """
    path = prepare_path(path)
    nomes = list(series)
    posicoes = np.arange(len(nomes))
    acertos = [series[n][0] for n in nomes]
    erros = [series[n][1] for n in nomes]

    figura, eixo = plt.subplots(figsize=(1.9 + 2.1 * len(nomes), 5.0))
    eixo.bar(posicoes - 0.2, acertos, width=0.4, color="tab:green", label="acertou")
    eixo.bar(posicoes + 0.2, erros, width=0.4, color="tab:red", label="errou")
    for i, (a, e) in enumerate(zip(acertos, erros)):
        eixo.annotate(
            f"separacao\n{a - e:+.3f}",
            (i, max(a, e) + 0.03),
            ha="center",
            fontsize=8,
        )
    if tau is not None:
        eixo.axhline(
            tau,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"limiar testado (tau={tau})",
        )
    eixo.set_xticks(posicoes)
    eixo.set_xticklabels(nomes)
    eixo.set_ylabel("confianca media do professor")
    eixo.set_ylim(0, 1.12)
    eixo.grid(axis="y", alpha=0.25)
    eixo.legend(fontsize=8)
    eixo.set_title(title)
    figura.tight_layout()
    figura.savefig(path, dpi=DPI)
    plt.close(figura)
    return path
