"""Figuras que nao pertencem a nenhuma trilha: curva de treino, matriz de confusao
e a curva de orcamento da Parte 2.

Separado de ``src.metrics`` de proposito: la ficam os numeros, e nada importa
matplotlib; aqui ficam as figuras. Quem so quer um F1 nao carrega backend grafico.

As figuras **de uma trilha** moram na pasta dela (``src/tracks/<trilha>/figures.py``),
e reusam ``prepare_path`` e ``DPI`` daqui. Apagar uma trilha leva embora as figuras
dela e nao toca neste arquivo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # sem display: roda no Colab, em CI e por ssh
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

DPI = 130


def prepare_path(path: Path | str) -> Path:
    """Cria o diretorio da figura e devolve o caminho. Publico: as trilhas reusam."""
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
    path = prepare_path(path)
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
    path = prepare_path(path)
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


def plot_budget_curve(
    series: Mapping[str, Sequence[tuple[int, float, float, int]]],
    path: Path | str,
    *,
    ceiling: float | None = None,
    references: Mapping[str, float] | None = None,
    title: str = "Curva de orcamento",
) -> Path:
    """F1 macro contra numero de rotulos — a figura obrigatoria da Parte 2.

    Cada serie e um metodo, e cada ponto uma tupla
    ``(orcamento, f1_medio, f1_desvio, seeds)``. O eixo x e **logaritmico**, como o
    enunciado pede: de 1 para 5 e de 5 para 20 o orcamento quintuplica e quadruplica,
    e em escala linear os dois primeiros pontos ficariam colados.

    A barra de erro aparece **so onde ha duas ou mais seeds**. Nos orcamentos de 1 e
    20 o protocolo pede uma seed, e desenhar barra zero ali sugeriria uma
    estabilidade que nao foi medida — o leitor veria um ponto exato onde ha uma
    amostra. O marcador vazado sinaliza esses pontos.

    ``references`` desenha linhas horizontais extra, uma por nome. Serve para o que
    **nao depende do orcamento**: na Trilha C o enunciado manda incluir o professor
    na curva, e o professor e zero-shot — ele nao usa rotulo nenhum, logo nao e um
    ponto que se move com o eixo x, e sim uma reta. Desenha-lo como curva sugeriria,
    falsamente, que ele melhora com mais rotulos.
    """
    path = prepare_path(path)
    figura, eixo = plt.subplots(figsize=(7.4, 5.2))

    if ceiling is not None:
        eixo.axhline(
            ceiling,
            color="tab:gray",
            linestyle="--",
            linewidth=1.2,
            label=f"teto de referencia ({ceiling:.3f})",
        )

    # cores distintas das series, para que as retas nao se confundam com as curvas
    for indice, (nome, valor) in enumerate(sorted((references or {}).items())):
        eixo.axhline(
            valor,
            color=f"C{6 + indice % 3}",
            linestyle=":",
            linewidth=1.4,
            label=f"{nome} ({valor:.3f})",
        )

    orcamentos: set[int] = set()
    for nome, pontos in series.items():
        pontos = sorted(pontos)
        x = [p[0] for p in pontos]
        y = [p[1] for p in pontos]
        # desvio so onde ha repeticao; nan nao desenha barra
        erro = [p[2] if p[3] > 1 else np.nan for p in pontos]
        orcamentos.update(x)
        linha = eixo.errorbar(
            x, y, yerr=erro, marker="o", capsize=4, linewidth=1.8, label=nome
        )
        cor = linha.lines[0].get_color()
        uma_seed = [(p[0], p[1]) for p in pontos if p[3] < 2]
        if uma_seed:
            eixo.scatter(
                [p[0] for p in uma_seed],
                [p[1] for p in uma_seed],
                facecolors="white",
                edgecolors=cor,
                zorder=3,
                s=45,
            )

    eixo.set_xscale("log")
    eixo.set_xticks(sorted(orcamentos))
    # sem isto o eixo log rotula 1, 5 e 20 em notacao cientifica
    eixo.get_xaxis().set_major_formatter(ScalarFormatter())
    eixo.minorticks_off()
    eixo.set_xlabel("rotulos por classe (escala log)")
    eixo.set_ylabel("F1 macro no test_public")
    eixo.set_title(title)
    eixo.grid(alpha=0.25)
    eixo.legend(fontsize=8)
    figura.tight_layout()
    figura.savefig(path, dpi=DPI)
    plt.close(figura)
    return path


# ===================== Parte 3 — galeria de falhas e atalhos ================
# Estas duas funcoes nao sabem o que e um erro, o que e um atalho nem de onde vem o
# mapa: recebem imagem, mapa e texto ja prontos. A analise fica em
# `src.analysis.saliency` e `src.analysis.gallery`, e e por isso que a mesma figura
# serve para a galeria de erros e para a de acertos.


@dataclass(frozen=True)
class SaliencyPanel:
    """Um mapa sobreposto a imagem, dentro de uma linha da galeria.

    ``region`` e a mascara da regiao atendida. Quando informada, e desenhada como
    contorno branco — e deve ser **a mesma** mascara usada no teste de oclusao, para
    que o contorno na figura seja literalmente a regiao cuja remocao foi medida, e nao
    uma ilustracao parecida.
    """

    label: str
    heatmap: NDArray[np.float64]
    region: NDArray[np.bool_] | None = None


@dataclass(frozen=True)
class SaliencyRow:
    """Uma linha da galeria: a imagem que o modelo viu e os mapas dela.

    ``image`` e a imagem **ja redimensionada** para a entrada da rede. Mostrar o
    original seria mostrar o que o modelo nao viu: o enunciado fala de "40x30 pixels
    depois do redimensionamento para 224", e a unica figura em que essa conta se
    verifica e a da imagem redimensionada.
    """

    image: NDArray[np.uint8]
    caption: str
    panels: tuple[SaliencyPanel, ...]


def plot_saliency_gallery(
    rows: Sequence[SaliencyRow],
    path: Path | str,
    *,
    title: str = "Galeria de saliencia",
) -> Path:
    """Uma linha por imagem: a imagem e os mapas, com a regiao atendida contornada.

    Escala de cor ``inferno`` e nao ``jet``: ``jet`` tem luminancia nao monotona e
    inventa fronteiras onde o mapa e suave, o que numa figura cujo proposito e
    "onde esta a evidencia" produz exatamente a leitura errada.
    """
    path = prepare_path(path)
    if not rows:
        raise ValueError("galeria vazia: nada a desenhar")
    colunas = 1 + max(len(linha.panels) for linha in rows)
    # layout="constrained" e nao tight_layout: as legendas de cada linha tem tres
    # linhas de texto e a barra de cor e criada depois dos eixos. Com tight_layout o
    # titulo de uma linha invade a imagem da linha de cima.
    figura, eixos = plt.subplots(
        len(rows),
        colunas,
        figsize=(3.3 * colunas, 3.8 * len(rows)),
        squeeze=False,
        layout="constrained",
    )
    imagem_mapeada = None
    for i, linha in enumerate(rows):
        eixos[i][0].imshow(linha.image)
        eixos[i][0].set_title(fill(linha.caption, 38), fontsize=8, loc="left")
        for j, painel in enumerate(linha.panels, start=1):
            eixo = eixos[i][j]
            eixo.imshow(linha.image)
            imagem_mapeada = eixo.imshow(
                painel.heatmap, cmap="inferno", alpha=0.55, vmin=0, vmax=1
            )
            if painel.region is not None:
                eixo.contour(
                    painel.region.astype(float),
                    levels=[0.5],
                    colors="white",
                    linewidths=0.9,
                )
            eixo.set_title(fill(painel.label, 38), fontsize=8, loc="left")
        for eixo in eixos[i]:
            eixo.set_xticks([])
            eixo.set_yticks([])
        # colunas vazias quando uma linha tem menos paineis que outra
        for j in range(1 + len(linha.panels), colunas):
            eixos[i][j].axis("off")

    figura.suptitle(title, fontsize=11)
    if imagem_mapeada is not None:
        barra = figura.colorbar(
            imagem_mapeada,
            ax=eixos.ravel().tolist(),
            orientation="horizontal",
            fraction=0.03,
            pad=0.02,
        )
        barra.set_label("evidencia relativa ao pico do proprio mapa", fontsize=8)
    figura.savefig(path, dpi=DPI)
    plt.close(figura)
    return path


@dataclass(frozen=True)
class MeasurePanel:
    """Uma medida comparada entre dois grupos de imagens.

    ``annotation`` e montada por quem chama, e nao aqui, porque ela carrega a leitura
    controlada pela classe — que e analise, nao desenho.
    """

    title: str
    group_names: tuple[str, str]
    group_values: tuple[NDArray[np.float64], NDArray[np.float64]]
    annotation: str = ""


def plot_measure_distributions(
    panels: Sequence[MeasurePanel],
    path: Path | str,
    *,
    title: str = "Medidas do mapa: acertos contra erros",
) -> Path:
    """Histogramas normalizados de uma medida nos dois grupos, com as medias marcadas.

    **Normalizados em densidade**, nao em contagem: no ``test_public`` os acertos sao
    varias vezes mais numerosos que os erros, e dois histogramas em contagem fariam o
    grupo menor desaparecer. O que a figura compara e a forma da distribuicao.
    """
    path = prepare_path(path)
    if not panels:
        raise ValueError("nenhuma medida para comparar")
    figura, eixos = plt.subplots(
        1, len(panels), figsize=(5.2 * len(panels), 4.2), squeeze=False
    )
    for eixo, painel in zip(eixos[0], panels):
        for k, (nome, valores) in enumerate(
            zip(painel.group_names, painel.group_values)
        ):
            if not len(valores):
                continue
            eixo.hist(
                valores,
                bins=24,
                density=True,
                alpha=0.45,
                color=f"C{k}",
                label=f"{nome} (n={len(valores)})",
            )
            eixo.axvline(
                float(np.mean(valores)), color=f"C{k}", linestyle="--", linewidth=1.4
            )
        eixo.set_title(painel.title, fontsize=10)
        eixo.set_ylabel("densidade")
        eixo.grid(alpha=0.2)
        eixo.legend(fontsize=8, loc="upper right")
        if painel.annotation:
            eixo.annotate(
                fill(painel.annotation, 30),
                xy=(0.02, 0.98),
                xycoords="axes fraction",
                va="top",
                fontsize=7.5,
                bbox={"boxstyle": "round", "fc": "white", "ec": "0.7", "alpha": 0.85},
            )
    figura.suptitle(title, fontsize=11)
    figura.tight_layout(rect=(0, 0, 1, 0.95))
    figura.savefig(path, dpi=DPI)
    plt.close(figura)
    return path
