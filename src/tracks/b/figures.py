"""Figuras dos dois diagnosticos obrigatorios da trilha B.

``plot_pseudo_label_curves``  precisao e cobertura por classe contra tau (obrig. 1)
``plot_label_distribution``   para onde vao os pseudo-rotulos (obrig. 2)

Reusam ``prepare_path`` e ``DPI`` de ``src.analysis.figures``, que e infraestrutura
comum. Nao importam nada de outra trilha.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # sem display: roda no Colab, em CI e por ssh
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.analysis.figures import DPI, prepare_path  # noqa: E402


def plot_pseudo_label_curves(
    curve,
    labels: Sequence[str],
    path: Path | str,
    *,
    title: str = "Pseudo-rotulos em funcao de tau",
) -> Path:
    """Precisao e cobertura **por classe** contra tau — entregavel obrigatorio 1.

    Dois mapas de calor (classe x tau) em vez de 21 linhas sobrepostas: com 21
    classes, um grafico de linhas vira emaranhado e nao se le classe nenhuma. O
    painel de baixo resume o que vale para a escolha de tau — quantas das 21
    classes ainda recebem algum pseudo-rotulo, e a que preco em precisao.
    """
    path = prepare_path(path)
    taus = list(curve.taus)
    ordem = np.argsort(-curve.support)  # classes frequentes no topo
    nomes = [labels[i] for i in ordem]

    figura, eixos = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        height_ratios=[3, 1],
        gridspec_kw={"hspace": 0.28, "wspace": 0.22},
    )

    for coluna, (matriz, nome) in enumerate(
        ((curve.precision, "precisao"), (curve.coverage, "cobertura"))
    ):
        eixo = eixos[0, coluna]
        imagem = eixo.imshow(
            matriz[:, ordem].T, cmap="Blues", vmin=0, vmax=1, aspect="auto"
        )
        eixo.set_xticks(range(len(taus)))
        eixo.set_xticklabels([f"{t:.2f}" for t in taus], fontsize=8)
        eixo.set_yticks(range(len(nomes)))
        eixo.set_yticklabels(
            [f"{n}  (n={int(curve.support[i])})" for n, i in zip(nomes, ordem)],
            fontsize=7,
        )
        eixo.set_xlabel("tau")
        eixo.set_title(f"{nome} por classe")
        figura.colorbar(imagem, ax=eixo, fraction=0.04)
        for linha in range(matriz.shape[0]):
            for coluna_classe, indice in enumerate(ordem):
                valor = matriz[linha, indice]
                if valor > 0:
                    eixo.text(
                        linha,
                        coluna_classe,
                        f"{valor:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="white" if valor > 0.5 else "black",
                    )

    resumo = eixos[1, 0]
    resumo.plot(
        taus,
        curve.overall_precision,
        "o-",
        color="tab:blue",
        label="precisao dos pseudo-rotulos emitidos",
    )
    resumo.plot(
        taus,
        curve.mask_rate,
        "s-",
        color="tab:orange",
        label="cobertura (fracao do conjunto)",
    )
    resumo.set_xlabel("tau")
    resumo.set_ylim(0, 1)
    resumo.grid(alpha=0.25)
    resumo.legend(fontsize=8)
    resumo.set_title("agregado")

    vivas = eixos[1, 1]
    vivas.plot(taus, curve.alive_classes, "o-", color="tab:red")
    vivas.axhline(curve.num_classes, color="gray", linestyle="--", linewidth=1)
    vivas.annotate(
        f"{curve.num_classes} classes",
        xy=(taus[0], curve.num_classes),
        fontsize=7,
        color="gray",
        xytext=(2, -10),
        textcoords="offset points",
    )
    vivas.set_xlabel("tau")
    vivas.set_ylabel("classes com algum pseudo-rotulo")
    vivas.set_ylim(0, curve.num_classes + 1)
    vivas.grid(alpha=0.25)
    vivas.set_title("colapso de classe")

    figura.suptitle(title)
    figura.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figura)
    return path


def plot_label_distribution(
    series: Mapping[str, Sequence[int]],
    true_counts: Sequence[int],
    labels: Sequence[str],
    path: Path | str,
    *,
    title: str = "Para onde vao os pseudo-rotulos",
) -> Path:
    """Fatia dos pseudo-rotulos contra a fatia real — entregavel obrigatorio 2.

    Compara proporcoes, nao contagens: o numero de pseudo-rotulos muda com tau e
    com o modelo, e so a fatia e comparavel entre variantes. Classes ordenadas da
    mais frequente para a mais rara, que e o eixo em que o vies aparece.
    """
    path = prepare_path(path)
    reais = np.asarray(true_counts, dtype=float)
    fatia_real = reais / reais.sum() if reais.sum() else reais
    ordem = np.argsort(-reais)
    posicoes = np.arange(len(labels))

    figura, eixos = plt.subplots(
        1, len(series), figsize=(5.2 * len(series), 7.4), sharey=True
    )
    if len(series) == 1:
        eixos = [eixos]

    for eixo, (nome, contagens) in zip(eixos, series.items()):
        pseudo = np.asarray(contagens, dtype=float)
        fatia = pseudo / pseudo.sum() if pseudo.sum() else pseudo
        eixo.barh(
            posicoes + 0.2,
            fatia_real[ordem],
            height=0.4,
            color="tab:gray",
            label="fatia real",
        )
        eixo.barh(
            posicoes - 0.2,
            fatia[ordem],
            height=0.4,
            color="tab:blue",
            label="fatia dos pseudo-rotulos",
        )
        eixo.set_yticks(posicoes)
        eixo.set_yticklabels([labels[i] for i in ordem], fontsize=7)
        eixo.invert_yaxis()
        eixo.set_xlabel("fracao do total")
        eixo.set_title(f"{nome}\n{int(pseudo.sum())} pseudo-rotulos")
        eixo.grid(axis="x", alpha=0.25)
    eixos[0].legend(fontsize=8, loc="lower right")

    figura.suptitle(title)
    figura.tight_layout()
    figura.savefig(path, dpi=DPI)
    plt.close(figura)
    return path
