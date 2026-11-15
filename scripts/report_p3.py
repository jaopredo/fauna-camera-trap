"""Relatorio da Parte 3 — galeria de falhas e atalhos.

    python scripts/report_p3.py
    python scripts/report_p3.py --device cpu
    python scripts/report_p3.py --execucao outputs/p1c_filter_b5_s0

Produz em ``outputs/p3/``:

    erros.png          os 3 erros, com o mapa da classe prevista e o da verdadeira
    acertos.png        os 5 acertos, com o mapa da classe prevista
    galeria.md         verdadeiro, previsto, confianca, por que entrou e o diagnostico
    medidas.csv        todos os numeros crus de cada imagem da galeria
    atalho.md          a pergunta inversa respondida nas 811 imagens, nao nas 5
    distribuicoes.png  as medidas do mapa em acertos contra erros

**Qual modelo.** O enunciado pede "tres erros do **modelo final**". Em vez de nomear
uma execucao no codigo, o script procura todas as execucoes do orcamento e escolhe a de
melhor F1 macro de **validacao** — a mesma regra que define o modelo entregue. Duas
consequencias boas: trocar o modelo entregue nao exige editar este arquivo, e a Parte 3
nao cita trilha nenhuma, continuando valida se uma trilha for apagada.

**Integridade.** A galeria sai do ``test_public``, que e o que o enunciado manda
("tres erros do modelo final" so existe onde ha rotulo para saber que errou). Nenhuma
decisao e tomada com esses numeros: a execucao vem do ``val``, a epoca veio do ``val``
durante o treino, e os limiares do diagnostico saem da distribuicao das proprias
medidas. Este script nao chama nada de ``src.evaluation.selection``.

**O que este relatorio nao faz, e por que.** Ele nao descreve o que aparece nas fotos.
O ``CLAUDE.md`` proibe usar modelo fechado para classificar imagem do conjunto, e
"o mapa esta no tronco da arvore ao fundo" e um ato de classificacao. Entao o
diagnostico automatico se limita ao que se mede — noite, forma do mapa, oclusao — e a
coluna final da ``galeria.md`` fica em branco, para quem apresenta preencher olhando a
figura com o gabarito do pacote na mao.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.analysis.figures import (
    MeasurePanel,
    SaliencyPanel,
    SaliencyRow,
    plot_measure_distributions,
    plot_saliency_gallery,
)
from src.analysis.gallery import (
    AttentionReference,
    GalleryPick,
    diagnose,
    select_errors,
    select_hits,
)
from src.analysis.groups import GroupComparison, compare_measure
from src.analysis.images import NIGHT_SATURATION_THRESHOLD, mean_saturation
from src.analysis.saliency import (
    CamModel,
    Faithfulness,
    MapStatistics,
    check_decomposition,
    class_activation_maps,
    decompose,
    faithfulness,
    heatmaps,
    map_statistics,
    mask_iou,
    scan,
    top_mass_mask,
)
from src.data.augment import build_augmentation
from src.data.catalog import DataCatalog
from src.data.dataset import load_rgb
from src.data.splits import Split
from src.data.transforms import as_displayable
from src.errors import AppError
from src.evaluation.runs import (
    LoadedRun,
    best_validation_metric,
    eval_loader,
    load_run,
)
from src.models.checkpoint import BEST_NAME
from src.tracking import resolve_device, runs_at_budget

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAIDA = PROJECT_ROOT / "outputs" / "p3"

ORCAMENTO_PADRAO = 5

#: As medidas comparadas entre acertos e erros: nome do atributo e como chama-la.
#: ``concentration`` e ``border_mass`` descrevem a forma do mapa; ``saturation`` e a
#: imagem, e esta aqui porque "foto noturna" e a primeira suspeita do enunciado.
MEDIDAS = (
    ("concentration", "concentracao da atencao"),
    ("border_mass", "massa do mapa na borda"),
    ("saturation", "saturacao da imagem (proxy de noite)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relatorio da Parte 3 do PA3")
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--execucao",
        type=Path,
        default=None,
        help="forca uma execucao; o padrao e a de melhor F1 macro de validacao",
    )
    parser.add_argument("--orcamento", type=int, default=ORCAMENTO_PADRAO)
    parser.add_argument("--erros", type=int, default=3)
    parser.add_argument("--acertos", type=int, default=5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def escolher_execucao(raiz: Path, orcamento: int) -> Path:
    """A execucao de melhor F1 macro de **validacao** no orcamento — o modelo entregue.

    Imprime todos os candidatos, nao so o vencedor: a escolha fica auditavel, e quem
    ler a saida ve que a ordenacao e por metrica de validacao.
    """
    candidatos = runs_at_budget(raiz, budget=orcamento, requires=BEST_NAME)
    if not candidatos:
        raise AppError(
            f"nenhuma execucao com {BEST_NAME} em {raiz} no orcamento {orcamento}. "
            f"Rode `make parte0` ou `make parte1c` antes."
        )
    pontuadas: list[tuple[float, int, Path]] = []
    for run_dir in candidatos:
        try:
            f1, epoca = best_validation_metric(run_dir)
        except AppError as erro:
            print(f"  (ignorando {run_dir.name}: {erro})")
            continue
        pontuadas.append((f1, epoca, run_dir))
    if not pontuadas:
        raise AppError("nenhuma execucao tem metrica de validacao no history.json")
    for f1, epoca, run_dir in sorted(pontuadas, reverse=True):
        print(f"  {run_dir.name:28} val F1 {f1:.4f} (epoca {epoca})")
    melhor_f1, _epoca, melhor = max(pontuadas)
    print(f"  modelo final: {melhor.name} (val F1 {melhor_f1:.4f})")
    print("  (o test_public nao participou desta escolha)")
    return melhor


def identificadores(
    catalog: DataCatalog, split: Split, y_true: np.ndarray
) -> tuple[str, ...]:
    """Os ``image_id`` na ordem do loader, conferida contra os rotulos.

    A conferencia nao e paranoia barata: se a ordem do loader divergisse da ordem do
    CSV, a galeria mostraria a imagem errada ao lado do numero certo — um erro que
    produz figura plausivel e nao levanta excecao nenhuma.
    """
    exemplos = catalog.labeled(split)
    rotulos = np.array([e.label_index for e in exemplos], dtype=np.int64)
    if rotulos.size != y_true.size or not np.array_equal(rotulos, y_true):
        raise AppError(
            f"a ordem do loader nao casa com {split.value}.csv "
            f"({rotulos.size} rotulos no CSV, {y_true.size} no loader)"
        )
    return tuple(e.image_id for e in exemplos)


@dataclass(frozen=True)
class ItemDaGaleria:
    """Uma imagem da galeria com tudo o que foi medido nela. So guarda e formata."""

    pick: GalleryPick
    image_id: str
    imagem: np.ndarray
    statistics: MapStatistics
    probe: Faithfulness
    saturation: float
    diagnostico: str
    #: mapa e regiao da classe prevista
    calor_previsto: np.ndarray
    regiao_prevista: np.ndarray
    #: idem para a classe verdadeira; ``None`` nos acertos, em que seria o mesmo mapa
    calor_verdadeiro: np.ndarray | None
    iou: float | None

    def linha_csv(self, classes: Sequence[str]) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "verdadeira": classes[self.pick.true_index],
            "prevista": classes[self.pick.predicted_index],
            "acertou": self.pick.correct,
            "saturacao": round(self.saturation, 2),
            "noturna": self.saturation < NIGHT_SATURATION_THRESHOLD,
            "iou_verdadeira_vs_prevista": (
                "" if self.iou is None else round(self.iou, 4)
            ),
            **self.statistics.as_row(),
            **self.probe.as_row(),
            "por_que_entrou": self.pick.reason,
            "diagnostico": self.diagnostico,
        }

    def legenda(self, classes: Sequence[str]) -> str:
        if self.pick.correct:
            return (
                f"{self.image_id}\nacerto: {classes[self.pick.true_index]} "
                f"(confianca {self.pick.confidence:.2f})"
            )
        return (
            f"{self.image_id}\nverdadeiro: {classes[self.pick.true_index]}\n"
            f"previsto: {classes[self.pick.predicted_index]} "
            f"(confianca {self.pick.confidence:.2f})"
        )

    def linha_da_figura(self, classes: Sequence[str]) -> SaliencyRow:
        paineis = [
            SaliencyPanel(
                label=(
                    f"CAM da classe prevista ({classes[self.pick.predicted_index]})"
                ),
                heatmap=self.calor_previsto,
                region=self.regiao_prevista,
            )
        ]
        if self.calor_verdadeiro is not None:
            paineis.append(
                SaliencyPanel(
                    label=(
                        f"CAM da classe verdadeira ({classes[self.pick.true_index]})"
                    ),
                    heatmap=self.calor_verdadeiro,
                )
            )
        return SaliencyRow(
            image=self.imagem,
            caption=self.legenda(classes),
            panels=tuple(paineis),
        )


def medir_galeria(
    escolhas: Sequence[GalleryPick],
    *,
    run: LoadedRun,
    cam: CamModel,
    catalog: DataCatalog,
    ids: Sequence[str],
    referencia: AttentionReference,
    device: str,
    com_mapa_da_verdadeira: bool,
) -> list[ItemDaGaleria]:
    """Calcula mapa, medidas, oclusao e diagnostico das imagens escolhidas.

    Um unico forward de features para todas elas, e um unico lote de oclusao. Sao 8
    imagens: o custo e irrelevante, mas a forma em lote evita a dispersao de um
    forward por imagem, em que uma imagem poderia passar por um caminho diferente.
    """
    if not escolhas:
        return []
    lado = run.config.data.image_size
    transformacao = build_augmentation("eval", image_size=lado)
    pils = [
        load_rgb(catalog.image_path(Split.TEST_PUBLIC, ids[p.position]))
        for p in escolhas
    ]
    lote = torch.stack([transformacao(imagem) for imagem in pils])

    previstas = [p.predicted_index for p in escolhas]
    with torch.no_grad():
        features = cam.forward_features(lote.to(device))
        calor_previsto = heatmaps(
            class_activation_maps(features, cam.weight, previstas), size=lado
        )
        calor_verdadeiro = (
            heatmaps(
                class_activation_maps(
                    features, cam.weight, [p.true_index for p in escolhas]
                ),
                size=lado,
            )
            if com_mapa_da_verdadeira
            else None
        )
    sondas = faithfulness(
        run.model, lote, list(calor_previsto), previstas, device=device
    )

    itens: list[ItemDaGaleria] = []
    for i, escolha in enumerate(escolhas):
        mapa = calor_previsto[i].detach().cpu().numpy()
        regiao = top_mass_mask(mapa)
        medidas = map_statistics(mapa)
        mapa_verdadeiro = (
            calor_verdadeiro[i].detach().cpu().numpy()
            if calor_verdadeiro is not None
            else None
        )
        iou = (
            mask_iou(regiao, top_mass_mask(mapa_verdadeiro))
            if mapa_verdadeiro is not None
            else None
        )
        saturacao = mean_saturation(pils[i])
        itens.append(
            ItemDaGaleria(
                pick=escolha,
                image_id=ids[escolha.position],
                imagem=as_displayable(lote[i]),
                statistics=medidas,
                probe=sondas[i],
                saturation=saturacao,
                diagnostico=diagnose(
                    medidas,
                    referencia,
                    saturation=saturacao,
                    iou_true_vs_predicted=iou,
                    probe=sondas[i],
                ).as_text(),
                calor_previsto=mapa,
                regiao_prevista=regiao,
                calor_verdadeiro=mapa_verdadeiro,
                iou=iou,
            )
        )
    return itens


# ===================== o que vai para o disco ===============================


def escrever_galeria(
    caminho: Path,
    erros: Sequence[ItemDaGaleria],
    acertos: Sequence[ItemDaGaleria],
    *,
    classes: Sequence[str],
    referencia: AttentionReference,
    execucao: str,
    lado: int,
    grade: int,
) -> Path:
    linhas = [
        "# Parte 3 — galeria de falhas e atalhos",
        "",
        f"Modelo: `{execucao}`, escolhido pelo F1 macro de **validacao**. Mapas de",
        "saliencia por **CAM** (`docs/DL_20242_13_Unsupervised.pdf`, p. 80-82).",
        "",
        "Numa rede que termina em global average pooling seguido de uma camada linear",
        "— e a ResNet-18 termina assim — CAM e Grad-CAM dao o mesmo mapa, a menos de",
        "uma constante positiva, porque o peso do Grad-CAM se reduz ao peso da `fc`",
        "dividido pelo numero de posicoes. Nao e analogia: `tests/test_saliency.py`",
        "verifica a igualdade contra o autograd.",
        "",
        "**Resolucao.** A `layer4` devolve uma grade 7x7 para uma entrada 224x224:",
        'cada celula do mapa cobre 32x32 pixels. O mapa sustenta "a evidencia esta no',
        'canto, nao no centro" e **nao** sustenta "o mapa esta na pata do animal".',
        "",
        f"**Referencia dos limiares.** {referencia.describe()}. Uma etiqueta como",
        '"atencao difusa" significa *difusa em relacao a este conjunto*, nao em',
        "relacao a um numero escolhido a mao.",
        "",
        f"## Os {len(erros)} erros (`erros.png`)",
        "",
    ]
    linhas += tabela_de_itens(erros, classes=classes, com_iou=True)
    linhas += [
        "",
        "A coluna IoU compara a regiao que o mapa da classe **prevista** aponta com a",
        "que o mapa da classe **verdadeira** aponta. Ela separa dois modos de falha que",
        "a matriz de confusao junta:",
        "",
        "* **IoU alto** — o modelo olhou para o lugar certo e nomeou errado. E confusao",
        "  de aparencia entre classes parecidas; mais rotulo daquelas classes ajuda.",
        "* **IoU baixo** — o modelo olhou para outro lugar. E falha de localizacao, e",
        "  mais rotulo da mesma classe nao resolve: o modelo esta aprendendo contexto.",
        "",
        f"## Os {len(acertos)} acertos (`acertos.png`)",
        "",
        "A pergunta inversa do enunciado. Estes sao os acertos **mais confiantes** de",
        "classes distintas — de proposito o melhor caso, porque um acerto confiante",
        "obtido olhando para o fundo e o atalho mais perigoso que existe: ele some",
        "quando a camera muda. A resposta representativa esta em `atalho.md`.",
        "",
    ]
    linhas += tabela_de_itens(acertos, classes=classes, com_iou=False)
    linhas += [
        "",
        "## O que estas medidas nao dizem",
        "",
        "Nenhuma delas diz se o que esta sob o mapa e um animal. Isso exigiria a caixa",
        "do animal, e o pacote nao traz anotacao de deteccao — o enunciado descreve o",
        'dado como imagem e rotulo de especie. Por isso nao existe etiqueta "animal',
        'pequeno": o substituto honesto e "regiao usada pequena", que fala do mapa.',
        "",
        "A leitura visual e da apresentacao: a figura, o gabarito do pacote e os olhos",
        "de quem apresenta. O que este arquivo garante e que a leitura visual vai",
        "encontrar os numeros ja medidos do lado dela, e nao o contrario.",
        "",
    ]
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return caminho


def tabela_de_itens(
    itens: Sequence[ItemDaGaleria], *, classes: Sequence[str], com_iou: bool
) -> list[str]:
    cabecalho = ["Imagem", "Verdadeira", "Prevista", "Confianca"]
    if com_iou:
        cabecalho.append("IoU prev/verd")
    cabecalho += ["Regiao usada", "O mapa explica?", "Diagnostico medido"]
    linhas = [
        "| " + " | ".join(cabecalho) + " |",
        "|" + "|".join(["---"] * len(cabecalho)) + "|",
    ]
    for item in itens:
        celulas = [
            f"`{item.image_id}`",
            classes[item.pick.true_index],
            classes[item.pick.predicted_index],
            f"{item.pick.confidence:.3f}",
        ]
        if com_iou:
            celulas.append("—" if item.iou is None else f"{item.iou:.2f}")
        celulas += [
            f"~{item.statistics.hot_side:.0f}x{item.statistics.hot_side:.0f} px",
            (
                f"sim ({item.probe.drop_hot():+.3f} contra "
                f"{item.probe.drop_cold():+.3f} do controle)"
                if item.probe.explains()
                else (
                    f"nao ({item.probe.drop_hot():+.3f} contra "
                    f"{item.probe.drop_cold():+.3f} do controle)"
                )
            ),
            item.diagnostico,
        ]
        linhas.append("| " + " | ".join(celulas) + " |")
    return linhas


def escrever_medidas(
    caminho: Path, itens: Sequence[ItemDaGaleria], *, classes: Sequence[str]
) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    linhas = [item.linha_csv(classes) for item in itens]
    with caminho.open("w", newline="", encoding="utf-8") as handle:
        escritor = csv.DictWriter(handle, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)
    return caminho


def escrever_atalho(
    caminho: Path,
    comparacoes: Sequence[GroupComparison],
    *,
    acertos: int,
    erros: int,
    area_da_borda: float,
) -> Path:
    """A pergunta inversa respondida no conjunto inteiro, com a classe controlada."""
    linhas = [
        "# Parte 3 — o modelo olha para o animal ou para o fundo?",
        "",
        "Cinco acertos nao respondem isso: eles ilustram. Aqui a mesma medida do mapa",
        f"e comparada entre os {acertos} acertos e os {erros} erros do `test_public`",
        "inteiro, e sempre em duas versoes.",
        "",
        "**Por que duas versoes.** Os erros se concentram nas classes dificeis. Se as",
        "classes dificeis tiverem mapas diferentes por qualquer outro motivo, a media",
        'bruta diz "erros tem mapa diferente" quando o que ela mediu foi "classes',
        'diferentes tem mapa diferente". A coluna controlada compara acertos contra',
        "erros **dentro de cada classe** com suporte nos dois lados, e e ela que",
        "responde. Este projeto ja levou esse golpe uma vez: ver `docs/decisions.md`,",
        "D19, em que a hipotese da noite inverteu de sinal ao controlar a classe.",
        "",
        "| Medida | Nos acertos | Nos erros | Bruto | **Controlando a classe** | Classes comparaveis |",
        "|---|---|---|---|---|---|",
    ]
    for c in comparacoes:
        marca = " ⚠" if c.confounded() else ""
        linhas.append(
            f"| {c.measure} | {c.raw_means[0]:.4f} (n={c.raw_counts[0]}) | "
            f"{c.raw_means[1]:.4f} (n={c.raw_counts[1]}) | {c.raw_delta():+.4f} | "
            f"**{c.controlled_text()}**{marca} | "
            f"{c.comparable_classes} de {c.total_classes} |"
        )
    linhas += [
        "",
        "O ⚠ marca as medidas em que o bruto e o controlado apontam para lados",
        "opostos — ou seja, em que a leitura ingenua estaria errada.",
        "",
        "## Como ler cada medida",
        "",
        "* **concentracao da atencao** — fracao da massa do mapa nos 10% de pixels mais",
        "  quentes. Um mapa uniforme daria 0,10. Mais alto quer dizer evidencia",
        "  localizada; mais baixo, evidencia espalhada pelo quadro, que e a assinatura",
        "  de decisao por contexto e nao por objeto.",
        "* **massa do mapa na borda** — fracao da massa na faixa externa. A faixa ocupa",
        f"  {area_da_borda:.1%} da area, entao esse e o valor nulo: um mapa uniforme deixa",
        "  essa fracao na borda sem querer. Acima dele o mapa esta puxado para a",
        "  moldura, e animal centrado nao produz isso.",
        "* **saturacao da imagem** — proxy de foto noturna em infravermelho (abaixo de",
        f"  {NIGHT_SATURATION_THRESHOLD:.0f} na escala 0-255). Nao e medida do mapa: esta aqui porque",
        '  "noite" e a primeira suspeita que o enunciado levanta.',
        "",
        "## O que uma diferenca pequena aqui significa",
        "",
        "Que a forma do mapa, sozinha, nao separa acerto de erro — e nao que o modelo",
        "olhe para o animal. As duas coisas sao diferentes, e distinguir uma da outra",
        "exigiria a caixa do animal, que o pacote nao tem. O que se pode afirmar com",
        "estes numeros e sobre **onde** o modelo se baseou, nunca sobre **o que** estava",
        "la.",
        "",
    ]
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return caminho


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    SAIDA.mkdir(parents=True, exist_ok=True)

    print("##### escolha do modelo final (no val)")
    run_dir = args.execucao or escolher_execucao(args.outputs, args.orcamento)
    execucao = load_run(run_dir, device=device)
    classes = execucao.classes
    lado = execucao.config.data.image_size
    print(
        f"  {run_dir.name} | {execucao.num_parameters():,} parametros | "
        f"entrada {lado}x{lado}"
    )

    cam = decompose(execucao.model)
    loader = eval_loader(execucao, Split.TEST_PUBLIC)

    print("\n##### varredura do test_public (um forward; mapa da classe prevista)")
    exemplo = next(iter(loader))[0][:2].to(device)
    diferenca = check_decomposition(execucao.model, cam, exemplo)
    with torch.no_grad():
        grade = int(cam.forward_features(exemplo).shape[-1])
    print(f"  a decomposicao reproduz o modelo (diferenca maxima {diferenca:.2e})")
    print(f"  grade do CAM: {grade}x{grade}, cada celula cobre {lado // grade} px")

    varredura = scan(cam, loader, device=device, size=lado)
    ids = identificadores(execucao.catalog, Split.TEST_PUBLIC, varredura.y_true)
    acertou = varredura.correct()
    referencia = AttentionReference.from_scan(varredura)
    print(
        f"  {len(ids)} imagens | {int(acertou.sum())} acertos, "
        f"{int((~acertou).sum())} erros"
    )
    print(f"  {referencia.describe()}")

    print("\n##### galeria")
    erros = medir_galeria(
        select_errors(
            varredura.probabilities,
            varredura.y_true,
            class_names=classes,
            limit=args.erros,
        ),
        run=execucao,
        cam=cam,
        catalog=execucao.catalog,
        ids=ids,
        referencia=referencia,
        device=device,
        com_mapa_da_verdadeira=True,
    )
    acertos = medir_galeria(
        select_hits(
            varredura.probabilities,
            varredura.y_true,
            class_names=classes,
            limit=args.acertos,
        ),
        run=execucao,
        cam=cam,
        catalog=execucao.catalog,
        ids=ids,
        referencia=referencia,
        device=device,
        com_mapa_da_verdadeira=False,
    )
    if not erros or not acertos:
        raise AppError(
            f"galeria incompleta: {len(erros)} erros e {len(acertos)} acertos. "
            f"O modelo nao errou ou nao acertou o suficiente no test_public?"
        )

    for item in (*erros, *acertos):
        print(
            f"  {item.image_id:28} {classes[item.pick.true_index]:22} -> "
            f"{classes[item.pick.predicted_index]:22} conf {item.pick.confidence:.3f}"
        )
        print(f"    {item.diagnostico}")

    plot_saliency_gallery(
        [item.linha_da_figura(classes) for item in erros],
        SAIDA / "erros.png",
        title=f"Parte 3 — tres erros de {run_dir.name} (CAM, test_public)",
    )
    plot_saliency_gallery(
        [item.linha_da_figura(classes) for item in acertos],
        SAIDA / "acertos.png",
        title=f"Parte 3 — cinco acertos de {run_dir.name} (CAM, test_public)",
    )
    escrever_galeria(
        SAIDA / "galeria.md",
        erros,
        acertos,
        classes=classes,
        referencia=referencia,
        execucao=run_dir.name,
        lado=lado,
        grade=grade,
    )
    escrever_medidas(SAIDA / "medidas.csv", (*erros, *acertos), classes=classes)

    print(f"\n##### a pergunta inversa, nas {len(ids)} imagens do test_public")
    saturacoes = np.array(
        [
            mean_saturation(load_rgb(execucao.catalog.image_path(Split.TEST_PUBLIC, i)))
            for i in ids
        ]
    )
    valores = {
        "concentration": varredura.measure("concentration"),
        "border_mass": varredura.measure("border_mass"),
        "saturation": saturacoes,
    }
    comparacoes = [
        compare_measure(
            valores[atributo],
            varredura.y_true,
            acertou,
            class_names=classes,
            group_names=("acertos", "erros"),
            measure=nome,
        )
        for atributo, nome in MEDIDAS
    ]
    for c in comparacoes:
        marca = "  ⚠ o bruto inverte ao controlar a classe" if c.confounded() else ""
        print(
            f"  {c.measure:34} bruto {c.raw_delta():+.4f} | "
            f"controlado {c.controlled_text()}{marca}"
        )

    plot_measure_distributions(
        [
            MeasurePanel(
                title=nome,
                group_names=("acertos", "erros"),
                group_values=(
                    valores[atributo][acertou],
                    valores[atributo][~acertou],
                ),
                annotation=(
                    f"bruto {c.raw_delta():+.4f}\n"
                    f"controlando a classe {c.controlled_text()}\n"
                    f"({c.comparable_classes} de {c.total_classes} classes comparaveis)"
                ),
            )
            for (atributo, nome), c in zip(MEDIDAS, comparacoes)
        ],
        SAIDA / "distribuicoes.png",
    )
    escrever_atalho(
        SAIDA / "atalho.md",
        comparacoes,
        acertos=int(acertou.sum()),
        erros=int((~acertou).sum()),
        area_da_borda=erros[0].statistics.border_area,
    )

    print(f"\n  artefatos: {SAIDA}")


if __name__ == "__main__":
    try:
        main()
    except AppError as erro:
        raise SystemExit(f"erro: {erro}")
