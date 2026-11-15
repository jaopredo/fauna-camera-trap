"""O professor zero-shot da trilha C: um foundation model de pesos abertos.

**O que o enunciado permite, literalmente.** "foundation models, isto e, CLIP,
SigLIP, BioCLIP (...). So na Trilha C, so com pesos abertos rodando no Colab ou na
maquina de voces, e so como professor: ele pode olhar o train e o test_public (para
voces medirem o professor), nunca o val durante o treino, e nao pode fazer parte do
modelo entregue."

Cada condicao tem uma contraparte no codigo:

* *pesos abertos, na sua maquina* — ``open_clip`` carrega do cache local do
  HuggingFace; nenhuma chamada de rede durante o treino do aluno;
* *so como professor* — este modulo **nao** expoe nada que o laco de treino consuma.
  Ele grava um arquivo e sai de cena (``TeacherCache``);
* *nunca o val* — ``cache_split`` recusa ``Split.VAL``. Nao e disciplina, e um
  ``TeacherPermissionError``;
* *nao faz parte do modelo entregue* — estrutural: o aluno le um ``.npz``, nao um
  modelo. ``predict.py`` nao tem como alcancar o professor.

**Por que precisa de codificador de texto.** O CLIP nao tem cabeca de classificacao:
ele foi treinado para por imagens e frases no mesmo espaco. A "cabeca" das 21 classes
e construida dos **nomes**, nao aprendida de imagens — e por isso que isto e
zero-shot e nao usa nenhum dos 105 rotulos:

    score(c) = <v, t_c> / (|v| |t_c|),  t_c = TextEncoder("a photo of a <nome>.")

**Fora do material da disciplina.** CLIP, SigLIP e BioCLIP nao aparecem nas aulas
11-15; destilacao professor-aluno tambem nao. Busca por "clip", "siglip", "bioclip" e
"contrastive" nas aulas 12 e 15: a 15 cobre ViT, Swin, MAE, MSN, SAM e visual
prompting, e a 12 cobre redes siamesas e perda contrastiva, nao language-image. O
enunciado nomeia estes modelos e os autoriza; as aulas nao os ensinam. Ver
``docs/decisions.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from src.data.splits import Split
from src.errors import AppError

#: Template do prompt, **fixo de antemao**. Nao escolhemos entre templates: isso
#: seria selecao de modelo, e o professor nao pode olhar o ``val``, enquanto o
#: ``test_public`` e proibido para escolha. Este e o template padrao documentado do
#: CLIP, e o mesmo para os dois professores, para que a comparacao isole o
#: pre-treino e nao o prompt.
PROMPT_TEMPLATE = "a photo of a {}."


class TeacherError(AppError, RuntimeError):
    """Falha ao construir ou rodar o professor."""


class TeacherPermissionError(AppError, PermissionError):
    """Tentativa de mostrar ao professor um conjunto que ele nao pode ver."""


#: O professor pode ver o ``train`` (para rotular) e o ``test_public`` (para medirmos
#: o professor, como o enunciado pede). O ``val`` e nosso, nao dele.
TEACHER_ALLOWED_SPLITS = (Split.TRAIN, Split.TEST_PUBLIC)


@dataclass(frozen=True)
class TeacherSpec:
    """Qual foundation model, com quais pesos abertos.

    ``label`` e o nome curto que aparece nas tabelas e no nome do arquivo de cache.
    """

    label: str
    model: str
    pretrained: str | None = None

    def describe(self) -> str:
        if self.pretrained:
            return f"{self.model} ({self.pretrained})"
        return self.model


#: Os dois professores. **Mesma arquitetura (ViT-B/16), pre-treino diferente** — e o
#: que torna a comparacao limpa: a diferenca mede o pre-treino no dominio, nao o
#: tamanho do modelo. BioCLIP foi treinado no TreeOfLife-10M (biologia); o CLIP da
#: OpenAI, em pares imagem-texto da web.
TEACHERS: dict[str, TeacherSpec] = {
    "bioclip": TeacherSpec(label="bioclip", model="hf-hub:imageomics/bioclip"),
    # `-quickgelu`, nao `ViT-B-16`: os pesos da OpenAI foram treinados com QuickGELU,
    # e o open_clip tem arquiteturas separadas para isso. Com o nome sem sufixo ele
    # carrega os pesos numa rede de GELU padrao e **avisa**, mas roda — produzindo um
    # F1 silenciosamente degradado. Medimos 0,384 com a ativacao errada antes de
    # notar o aviso; e o tipo de erro que viraria um numero plausivel no relatorio.
    "clip": TeacherSpec(label="clip", model="ViT-B-16-quickgelu", pretrained="openai"),
}


def available_teachers() -> tuple[str, ...]:
    return tuple(sorted(TEACHERS))


def resolve_teacher(name: str) -> TeacherSpec:
    try:
        return TEACHERS[name]
    except KeyError:
        raise TeacherError(
            f"professor desconhecido: {name!r}. Disponiveis: {list(available_teachers())}"
        ) from None


def ensure_teacher_may_see(split: Split) -> Split:
    """Barreira do ``val``, no mesmo espirito de ``src.data.splits.ensure_selectable``.

    O enunciado diz que o professor nunca pode olhar o ``val``. Aqui isso e uma
    excecao, nao um comentario: nenhuma funcao deste modulo aceita o ``val``, e o
    teste de contrato enumera as funcoes publicas para garantir que continue assim.
    """
    if split not in TEACHER_ALLOWED_SPLITS:
        raise TeacherPermissionError(
            f"o professor nao pode ver o {split.value}: o enunciado libera apenas "
            f"{[s.value for s in TEACHER_ALLOWED_SPLITS]}. O val e so nosso, para "
            f"selecao de modelo."
        )
    return split


def class_prompts(
    class_names: Sequence[str], *, template: str = PROMPT_TEMPLATE
) -> tuple[str, ...]:
    """Uma frase por classe, a partir do nome em ``classes.csv``.

    ``american_black_bear`` vira ``"a photo of a american black bear."``. O
    sublinhado e separador de arquivo, nao parte do nome da especie.
    """
    return tuple(template.format(nome.replace("_", " ")) for nome in class_names)


def load_teacher(spec: TeacherSpec, *, device: str):
    """Carrega o modelo, o tokenizador e o pre-processamento **dele**.

    Usamos o pre-processamento documentado do proprio professor, nao o nosso
    (``src/data/transforms.py``). Dois motivos: as estatisticas de normalizacao do
    CLIP nao sao as do ImageNet, e o numero que reportamos tem de ser o F1 zero-shot
    *daquele modelo*, comparavel ao que qualquer um mediria.

    O preco e conhecido e entra no diagnostico: o pre-processamento do CLIP faz
    ``Resize`` do lado menor e ``CenterCrop``, que **descarta as bordas** — contra a
    nossa D9, que redimensiona o quadro inteiro justamente porque o animal pode
    ocupar 2% da imagem e estar em qualquer canto. Onde o professor erra em animais
    pequenos ou na borda, o recorte e suspeito numero um, e isso e material do
    diagnostico obrigatorio 2.
    """
    try:
        import open_clip
    except ImportError as erro:  # pragma: no cover - depende do ambiente
        raise TeacherError(
            "open_clip nao esta instalado. Rode `make setup-professor`. "
            "Ele fica fora do pyproject pelo mesmo motivo do torch (ver D11)."
        ) from erro

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            spec.model, pretrained=spec.pretrained
        )
        tokenizer = open_clip.get_tokenizer(spec.model)
    except Exception as erro:  # pragma: no cover - rede ou nome errado
        raise TeacherError(f"nao consegui carregar {spec.describe()}: {erro}") from erro
    return model.eval().to(device), tokenizer, preprocess


@torch.no_grad()
def build_zero_shot_head(
    model, tokenizer, class_names: Sequence[str], *, device: str
) -> torch.Tensor:
    """A cabeca de classificacao, construida das palavras: ``(num_classes, dim)``.

    Roda o codificador de texto 21 vezes. Custo computacional irrelevante — ele
    nao esta aqui por desempenho, esta porque sem ele nao existe cabeca nenhuma.
    """
    tokens = tokenizer(list(class_prompts(class_names))).to(device)
    texto = model.encode_text(tokens).float()
    return texto / texto.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def teacher_probabilities(
    model,
    head: torch.Tensor,
    batches: Iterable[tuple[torch.Tensor, Sequence[str]]],
    *,
    device: str,
) -> tuple[NDArray[np.float32], tuple[str, ...]]:
    """Probabilidades do professor para cada imagem: ``(n, num_classes)`` e os ids.

    A escala logit vem do proprio modelo: o CLIP a aprendeu no pre-treino, e e ela
    que torna o softmax calibrado em vez de quase uniforme. Ignora-la achataria
    todas as confiancas e quebraria qualquer filtro por limiar depois.
    """
    escala = float(model.logit_scale.exp().detach())
    partes: list[NDArray[np.float32]] = []
    ids: list[str] = []
    for imagens, lote_ids in batches:
        visual = model.encode_image(imagens.to(device)).float()
        visual = visual / visual.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        logits = escala * visual @ head.T
        partes.append(torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32))
        ids.extend(lote_ids)
    if not partes:
        raise TeacherError("nenhuma imagem para o professor")
    return np.concatenate(partes), tuple(ids)


@dataclass(frozen=True)
class TeacherCache:
    """O **unico** canal entre o professor e o aluno: um arquivo em disco.

    Esta fronteira e o que faz "o professor nao pode fazer parte do modelo
    entregue" ser estrutural. O aluno recebe uma matriz de probabilidades indexada
    por ``image_id``; ele nao sabe que modelo a produziu, nao consegue chama-lo, e
    ``predict.py`` nao tem caminho ate aqui.
    """

    teacher: str
    split: str
    image_ids: tuple[str, ...]
    probabilities: NDArray[np.float32]
    class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.probabilities.shape != (len(self.image_ids), len(self.class_names)):
            raise TeacherError(
                f"cache incoerente: {self.probabilities.shape} para "
                f"{len(self.image_ids)} imagens e {len(self.class_names)} classes"
            )

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def index(self) -> dict[str, int]:
        """``image_id`` -> linha. O aluno procura por id, nao por posicao."""
        return {image_id: i for i, image_id in enumerate(self.image_ids)}

    def hard_labels(self) -> NDArray[np.int64]:
        return self.probabilities.argmax(axis=1).astype(np.int64)

    def confidence(self) -> NDArray[np.float32]:
        return self.probabilities.max(axis=1)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            teacher=self.teacher,
            split=self.split,
            image_ids=np.array(self.image_ids),
            probabilities=self.probabilities,
            class_names=np.array(self.class_names),
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> "TeacherCache":
        path = Path(path)
        if not path.exists():
            raise TeacherError(
                f"cache do professor nao existe: {path}. "
                f"Rode `python scripts/teacher.py` antes de treinar o aluno."
            )
        with np.load(path, allow_pickle=False) as dados:
            return cls(
                teacher=str(dados["teacher"]),
                split=str(dados["split"]),
                image_ids=tuple(str(x) for x in dados["image_ids"]),
                probabilities=dados["probabilities"].astype(np.float32),
                class_names=tuple(str(x) for x in dados["class_names"]),
            )


def cache_path(root: Path | str, teacher: str, split: Split) -> Path:
    """Onde o cache daquele professor naquele conjunto fica."""
    return Path(root) / f"{teacher}_{split.value}.npz"


def iter_batches(
    loader: Iterable,
) -> Iterator[tuple[torch.Tensor, Sequence[str]]]:
    """Normaliza o que os dois loaders entregam para ``(imagens, ids)``.

    O ``UnlabeledImageDataset`` devolve ``(imagens, ids)`` e o rotulado
    ``(imagens, rotulos)``. O professor so precisa das imagens e de uma chave; quando
    a chave vem como tensor de rotulos, ela nao serve de id, e quem chama passa o
    loader sem rotulo.
    """
    for imagens, chaves in loader:
        if isinstance(chaves, torch.Tensor):
            raise TeacherError(
                "o professor precisa de image_ids, nao de rotulos: use o loader "
                "sem rotulo, que devolve (imagens, ids)"
            )
        yield imagens, list(chaves)
