"""Figuras dos diagnosticos. Recebem dados prontos e gravam PNG.

Separado de ``src.metrics`` de proposito: la ficam os numeros, e nada importa
matplotlib; aqui ficam as figuras. Quem so quer um F1 nao carrega backend grafico.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # sem display: roda no Colab, em CI e por ssh
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 130


def _prepare(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_training_curve(
    history: Sequence[Mapping[str, object]],
    path: Path | str,
    *,
    title: str = "Treino e validacao",
    monitor: str = "macro_f1",
) -> Path:
    """Perda de treino e metrica de validacao por epoca, com a melhor epoca marcada.

    Responde ao item 3 da Parte 0: a distancia entre a perda de treino caindo para
    perto de zero e o F1 de validacao parando de subir e a memorizacao das 105
    imagens. A linha vertical marca a epoca escolhida no ``val`` — que e a que vai
    para o ``best.pt``.
    """
    path = _prepare(path)
    epocas = [int(linha["epoch"]) for linha in history]
    perda = [float(linha["train_loss"]) for linha in history]
    chave = f"val_{monitor}"
    metrica = [float(linha[chave]) for linha in history if linha.get(chave) is not None]

    figura, eixo_perda = plt.subplots(figsize=(8, 4.5))
    eixo_perda.plot(epocas, perda, color="tab:red", label="perda de treino")
    eixo_perda.set_xlabel("epoca")
    eixo_perda.set_ylabel("perda de treino", color="tab:red")
    eixo_perda.tick_params(axis="y", labelcolor="tab:red")
    eixo_perda.grid(alpha=0.25)

    if metrica:
        eixo_metrica = eixo_perda.twinx()
        eixo_metrica.plot(
            epocas[: len(metrica)],
            metrica,
            color="tab:blue",
            label=f"{monitor} (val)",
        )
        eixo_metrica.set_ylabel(f"{monitor} de validacao", color="tab:blue")
        eixo_metrica.tick_params(axis="y", labelcolor="tab:blue")
        melhor = int(np.argmax(metrica))
        eixo_metrica.axvline(
            epocas[melhor], color="tab:blue", linestyle="--", alpha=0.6
        )
        eixo_metrica.annotate(
            f"melhor epoca: {epocas[melhor]}\n{monitor} {metrica[melhor]:.3f}",
            xy=(epocas[melhor], metrica[melhor]),
            xytext=(6, -28),
            textcoords="offset points",
            fontsize=8,
            color="tab:blue",
        )

    eixo_perda.set_title(title)
    figura.tight_layout()
    figura.savefig(path, dpi=DPI)
    plt.close(figura)
    return path


def plot_confusion_matrix(
    confusion: Sequence[Sequence[int]],
    labels: Sequence[str],
    path: Path | str,
    *,
    title: str = "Matriz de confusao",
    normalize: bool = True,
) -> Path:
    """Matriz de confusao; linha = classe verdadeira, coluna = prevista.

    Normalizada por linha quando ``normalize``: com cavalo tendo 4 imagens no
    ``test_public`` e urso 82, a matriz em contagem absoluta so mostraria as classes
    grandes. Normalizar por linha faz cada celula virar "que fracao desta classe foi
    para aquela", que e a leitura que interessa.
    """
    path = _prepare(path)
    matriz = np.asarray(confusion, dtype=float)
    suporte = matriz.sum(axis=1, keepdims=True)
    exibida = (
        np.divide(matriz, suporte, out=np.zeros_like(matriz), where=suporte > 0)
        if normalize
        else matriz
    )

    figura, eixo = plt.subplots(figsize=(10, 8.6))
    imagem = eixo.imshow(exibida, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    figura.colorbar(
        imagem,
        ax=eixo,
        fraction=0.046,
        label="fracao da classe verdadeira" if normalize else "imagens",
    )

    eixo.set_xticks(range(len(labels)))
    eixo.set_yticks(range(len(labels)))
    eixo.set_xticklabels(labels, rotation=90, fontsize=7)
    eixo.set_yticklabels(
        [f"{nome}  (n={int(s)})" for nome, s in zip(labels, suporte.ravel())],
        fontsize=7,
    )
    eixo.set_xlabel("previsto")
    eixo.set_ylabel("verdadeiro")
    eixo.set_title(title)

    # anota so as celulas com peso, para a figura continuar legivel com 21 classes
    limite = 0.5 if normalize else matriz.max() / 2 if matriz.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            valor = exibida[i, j]
            if valor > 0:
                eixo.text(
                    j,
                    i,
                    f"{valor:.2f}" if normalize else f"{int(valor)}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if valor > limite else "black",
                )

    figura.tight_layout()
    figura.savefig(path, dpi=DPI)
    plt.close(figura)
    return path
