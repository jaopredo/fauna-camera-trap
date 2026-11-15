"""O professor da trilha C: a barreira do val, o cache e os prompts.

Nao carrega modelo nenhum: `open_clip` e instalado a parte e baixa ~600 MB. O que
estes testes protegem e a logica em volta — que e justamente onde as regras do
enunciado vivem.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.data.splits import Split
from src.tracks.c import teacher as modulo
from src.tracks.c.teacher import (
    PROMPT_TEMPLATE,
    TEACHER_ALLOWED_SPLITS,
    TeacherCache,
    TeacherError,
    TeacherPermissionError,
    available_teachers,
    cache_path,
    class_prompts,
    ensure_teacher_may_see,
    resolve_teacher,
)

K = 3
NOMES = ("american_black_bear", "bird", "horse")


def cache(n: int = 4) -> TeacherCache:
    probs = np.full((n, K), 1 / K, dtype=np.float32)
    probs[:, 0] = 0.8
    probs[:, 1:] = 0.1
    return TeacherCache(
        teacher="falso",
        split="train",
        image_ids=tuple(f"img{i}" for i in range(n)),
        probabilities=probs,
        class_names=NOMES,
    )


# ===================== a barreira do val ====================================


def test_o_professor_nao_pode_ver_o_val():
    """O enunciado: "nunca o val durante o treino". Excecao, nao comentario."""
    with pytest.raises(TeacherPermissionError, match="val"):
        ensure_teacher_may_see(Split.VAL)


@pytest.mark.parametrize("split", TEACHER_ALLOWED_SPLITS)
def test_o_professor_pode_ver_train_e_test_public(split):
    """O enunciado libera os dois: o train para rotular, o test_public para medi-lo."""
    assert ensure_teacher_may_see(split) is split


def test_o_val_nao_esta_na_lista_de_permitidos():
    assert Split.VAL not in TEACHER_ALLOWED_SPLITS


def test_toda_funcao_que_recebe_split_passa_pela_barreira():
    """Contrato por introspecao: uma funcao nova nao pode esquecer a barreira.

    Enumera as funcoes publicas do modulo que tem um parametro ``split`` e exige
    que cada uma ou chame ``ensure_teacher_may_see`` ou seja ela mesma / um
    utilitario de caminho. Sem isto, acrescentar uma funcao que aceita conjunto
    abriria um buraco silencioso na regra do enunciado.
    """
    isentas = {"ensure_teacher_may_see", "cache_path"}
    for nome, objeto in inspect.getmembers(modulo, inspect.isfunction):
        if nome.startswith("_") or objeto.__module__ != modulo.__name__:
            continue
        if nome in isentas:
            continue
        if "split" not in inspect.signature(objeto).parameters:
            continue
        fonte = inspect.getsource(objeto)
        assert "ensure_teacher_may_see" in fonte, (
            f"{nome} recebe um split e nao chama ensure_teacher_may_see: o val "
            f"passaria sem ser recusado"
        )


# ===================== prompts ==============================================


def test_o_prompt_usa_o_nome_da_classe_sem_sublinhado():
    """O sublinhado e separador de arquivo, nao parte do nome da especie."""
    prompts = class_prompts(NOMES)
    assert prompts[0] == "a photo of a american black bear."
    assert "_" not in prompts[0]


def test_ha_um_prompt_por_classe():
    assert len(class_prompts(NOMES)) == len(NOMES)


def test_o_template_e_fixo_no_modulo():
    """Nao e parametro de execucao de proposito: escolher template seria selecao,
    e o professor nao pode ver o val enquanto o test_public e proibido para isso."""
    assert "{}" in PROMPT_TEMPLATE


# ===================== os professores =======================================


def test_os_dois_professores_estao_declarados():
    assert set(available_teachers()) == {"bioclip", "clip"}


def test_os_dois_sao_a_mesma_arquitetura():
    """ViT-B/16 nos dois: a diferenca mede o pre-treino, nao o tamanho do modelo."""
    assert "B-16" in resolve_teacher("clip").model
    assert "bioclip" in resolve_teacher("bioclip").model


def test_os_pesos_openai_usam_quickgelu():
    """Sem o sufixo, o open_clip carrega os pesos numa rede de GELU padrao, avisa,
    e roda — produzindo um F1 silenciosamente degradado. Ja nos pegou uma vez."""
    assert resolve_teacher("clip").model.endswith("-quickgelu")


def test_professor_desconhecido_lista_as_opcoes():
    with pytest.raises(TeacherError, match="Disponiveis"):
        resolve_teacher("gpt")


# ===================== o cache, que e a fronteira ===========================


def test_o_cache_sobrevive_a_ida_e_volta(tmp_path):
    original = cache()
    recarregado = TeacherCache.load(original.save(tmp_path / "c.npz"))
    assert recarregado.image_ids == original.image_ids
    assert recarregado.class_names == original.class_names
    assert recarregado.teacher == original.teacher
    assert np.allclose(recarregado.probabilities, original.probabilities)


def test_o_cache_indexa_por_id_e_nao_por_posicao():
    """O aluno procura por image_id: confiar na ordem do loader e bug silencioso."""
    indice = cache().index()
    assert indice["img2"] == 2
    assert len(indice) == 4


def test_rotulo_duro_e_confianca_saem_das_probabilidades():
    c = cache()
    assert c.hard_labels().tolist() == [0, 0, 0, 0]
    assert c.confidence()[0] == pytest.approx(0.8)


def test_cache_ausente_diz_qual_script_rodar(tmp_path):
    with pytest.raises(TeacherError, match="scripts/teacher.py"):
        TeacherCache.load(tmp_path / "nao_existe.npz")


def test_forma_incoerente_e_recusada_na_construcao():
    with pytest.raises(TeacherError, match="incoerente"):
        TeacherCache(
            teacher="falso",
            split="train",
            image_ids=("a",),
            probabilities=np.zeros((2, K), dtype=np.float32),
            class_names=NOMES,
        )


def test_o_caminho_do_cache_separa_professor_e_conjunto(tmp_path):
    a = cache_path(tmp_path, "clip", Split.TRAIN)
    b = cache_path(tmp_path, "bioclip", Split.TRAIN)
    c = cache_path(tmp_path, "clip", Split.TEST_PUBLIC)
    assert len({a, b, c}) == 3, "dois professores nao podem gravar no mesmo arquivo"
