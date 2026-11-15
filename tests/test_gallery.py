"""A galeria da Parte 3: as regras de escolha das imagens e o diagnostico medido.

O que estes testes protegem nao e um numero, e uma propriedade de integridade: a
escolha das 8 imagens da galeria tem de ser **determinada pelos dados**, para que ela
nao possa ser ajustada depois de ver os mapas.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.gallery import (
    LOW_IOU,
    AttentionReference,
    GalleryError,
    diagnose,
    select_errors,
    select_hits,
)
from src.analysis.images import NIGHT_SATURATION_THRESHOLD
from src.analysis.saliency import Faithfulness, MapStatistics, SaliencyScan

NOMES = ("urso", "corvo", "passaro", "cavalo")


def probabilidades(pares):
    """Monta uma matriz de probabilidade a partir de ``(classe_prevista, confianca)``."""
    matriz = np.zeros((len(pares), len(NOMES)))
    for i, (prevista, confianca) in enumerate(pares):
        resto = (1.0 - confianca) / (len(NOMES) - 1)
        matriz[i] = resto
        matriz[i, prevista] = confianca
    return matriz


def medidas(**campos) -> MapStatistics:
    padrao = {
        "concentration": 0.40,
        "border_mass": 0.30,
        "border_area": 0.64,
        "peak_row": 100,
        "peak_col": 100,
        "hot_pixels": 5000,
        "side": 224,
    }
    return MapStatistics(**{**padrao, **campos})


# ===================== escolha dos erros ====================================


def test_os_erros_vem_um_por_par_de_confusao_e_na_ordem_de_frequencia():
    """Tres erros do mesmo par mostrariam um modo de falha tres vezes."""
    # par (0 -> 1) tres vezes, par (2 -> 3) duas vezes, par (3 -> 0) uma vez
    verdade = [0, 0, 0, 2, 2, 3, 1]
    probs = probabilidades(
        [(1, 0.6), (1, 0.9), (1, 0.7), (3, 0.8), (3, 0.5), (0, 0.95), (1, 0.99)]
    )
    escolhas = select_errors(probs, verdade, class_names=NOMES, limit=3)

    assert [(e.true_index, e.predicted_index) for e in escolhas] == [
        (0, 1),
        (2, 3),
        (3, 0),
    ]
    assert escolhas[0].position == 1, "dentro do par, o erro mais confiante"
    assert escolhas[0].confidence == pytest.approx(0.9)
    assert all("confusao mais frequente" in e.reason for e in escolhas)
    assert all(not e.correct for e in escolhas)


def test_menos_pares_que_o_limite_devolve_o_que_existe():
    probs = probabilidades([(1, 0.6), (0, 0.9)])
    escolhas = select_errors(probs, [0, 0], class_names=NOMES, limit=3)
    assert len(escolhas) == 1


def test_sem_erro_nenhum_a_galeria_de_erros_fica_vazia():
    probs = probabilidades([(0, 0.6), (1, 0.9)])
    assert select_errors(probs, [0, 1], class_names=NOMES) == ()


# ===================== escolha dos acertos ==================================


def test_os_acertos_sao_os_mais_confiantes_de_classes_distintas():
    """Com 82 ursos e 4 cavalos, os cinco mais confiantes seriam cinco ursos."""
    verdade = [0, 0, 0, 1, 2]
    probs = probabilidades([(0, 0.99), (0, 0.98), (0, 0.97), (1, 0.80), (2, 0.60)])
    escolhas = select_hits(probs, verdade, class_names=NOMES, limit=3)

    assert [e.true_index for e in escolhas] == [0, 1, 2]
    assert [e.position for e in escolhas] == [0, 3, 4]
    assert escolhas[0].confidence == pytest.approx(0.99)
    assert all(e.correct for e in escolhas)


def test_menos_classes_acertadas_que_o_limite_devolve_o_que_existe():
    probs = probabilidades([(0, 0.9), (0, 0.8)])
    assert len(select_hits(probs, [0, 0], class_names=NOMES, limit=5)) == 1


def test_a_escolha_nao_depende_da_ordem_das_imagens_empatadas():
    """Confianca igual: o desempate e pela posicao, logo o resultado e estavel."""
    probs = probabilidades([(0, 0.9), (1, 0.9)])
    primeira = select_hits(probs, [0, 1], class_names=NOMES, limit=2)
    segunda = select_hits(probs, [0, 1], class_names=NOMES, limit=2)
    assert [e.position for e in primeira] == [e.position for e in segunda] == [0, 1]


def test_tamanhos_incompativeis_sao_recusados():
    with pytest.raises(GalleryError):
        select_hits(probabilidades([(0, 0.9)]), [0, 1], class_names=NOMES)


# ===================== a referencia do diagnostico ==========================


def varredura(concentracoes, bordas=None, quentes=None) -> SaliencyScan:
    n = len(concentracoes)
    estatisticas = tuple(
        medidas(
            concentration=c,
            border_mass=0.3 if bordas is None else bordas[i],
            hot_pixels=5000 if quentes is None else quentes[i],
        )
        for i, c in enumerate(concentracoes)
    )
    return SaliencyScan(
        probabilities=np.full((n, len(NOMES)), 1 / len(NOMES)),
        y_true=np.zeros(n, dtype=np.int64),
        statistics=estatisticas,
    )


def test_a_referencia_sai_dos_percentis_do_conjunto():
    """ "Difuso" significa "abaixo da mediana destas imagens", nao de um numero a mao."""
    referencia = AttentionReference.from_scan(varredura([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert referencia.diffuse_below == pytest.approx(0.3)
    assert referencia.images == 5
    assert "mediana" in referencia.describe()
    assert referencia.describe().startswith("5 imagens")


def test_varredura_vazia_nao_produz_referencia():
    with pytest.raises(GalleryError):
        AttentionReference.from_scan(varredura([]))


# ===================== o diagnostico ========================================


REFERENCIA = AttentionReference(
    diffuse_below=0.30, border_above=0.70, small_region_below=1000.0, images=811
)


def sonda(fiel: bool) -> Faithfulness:
    """Fiel: apagar a regiao apontada custa mais que apagar o controle frio."""
    return Faithfulness(
        baseline=0.9,
        without_hot=0.3 if fiel else 0.88,
        without_cold=0.85 if fiel else 0.4,
        only_hot=0.8,
    )


def test_sem_nada_anormal_o_diagnostico_diz_isso_em_vez_de_inventar():
    d = diagnose(medidas(), REFERENCIA, saturation=120.0, probe=sonda(True))
    assert d.tags == ()
    assert "nada fora do padrao" in d.as_text()


def test_noite_entra_pelo_proxy_de_saturacao():
    d = diagnose(
        medidas(),
        REFERENCIA,
        saturation=NIGHT_SATURATION_THRESHOLD - 1,
        probe=sonda(True),
    )
    assert any("noturna" in etiqueta for etiqueta in d.tags)


def test_sem_saturacao_nao_ha_etiqueta_de_noite():
    d = diagnose(medidas(), REFERENCIA, probe=sonda(True))
    assert not any("noturna" in etiqueta for etiqueta in d.tags)


def test_atencao_difusa_e_relativa_a_mediana_do_conjunto():
    d = diagnose(medidas(concentration=0.2), REFERENCIA, probe=sonda(True))
    assert any("difusa" in etiqueta for etiqueta in d.tags)


def test_massa_na_borda_exige_tambem_estar_acima_da_area_da_borda():
    """Massa alta na borda nao e atalho se a borda ocupa mais area ainda."""
    alta = diagnose(
        medidas(border_mass=0.80, border_area=0.64), REFERENCIA, probe=sonda(True)
    )
    assert any("borda" in etiqueta for etiqueta in alta.tags)

    # acima do percentil do conjunto, mas abaixo da area que a faixa ocupa
    ambigua = diagnose(
        medidas(border_mass=0.75, border_area=0.90), REFERENCIA, probe=sonda(True)
    )
    assert not any("borda" in etiqueta for etiqueta in ambigua.tags)


def test_regiao_pequena_fala_do_mapa_e_nao_do_animal():
    d = diagnose(medidas(hot_pixels=400), REFERENCIA, probe=sonda(True))
    etiqueta = next(e for e in d.tags if "regiao usada pequena" in e)
    assert "20x20 px" in etiqueta, "o lado equivalente de 400 pixels e 20"
    assert "animal" not in etiqueta


def test_iou_baixo_vira_olhou_para_outro_lugar():
    d = diagnose(
        medidas(),
        REFERENCIA,
        iou_true_vs_predicted=LOW_IOU - 0.01,
        probe=sonda(True),
    )
    assert any("outro lugar" in etiqueta for etiqueta in d.tags)

    alto = diagnose(medidas(), REFERENCIA, iou_true_vs_predicted=0.9, probe=sonda(True))
    assert not any("outro lugar" in etiqueta for etiqueta in alto.tags)


def test_mapa_infiel_e_denunciado_pelo_proprio_diagnostico():
    """Se a oclusao desmente o mapa, o diagnostico tem de dizer — nao esconder."""
    d = diagnose(medidas(), REFERENCIA, probe=sonda(False))
    assert any("nao explica a decisao" in etiqueta for etiqueta in d.tags)


def test_todas_as_etiquetas_carregam_o_numero_que_as_justifica():
    """Etiqueta sem numero e opiniao; com numero, e leitura de medida."""
    d = diagnose(
        medidas(concentration=0.1, border_mass=0.8, hot_pixels=100),
        REFERENCIA,
        saturation=5.0,
        iou_true_vs_predicted=0.05,
        probe=sonda(False),
    )
    assert len(d.tags) == 6
    assert all(any(c.isdigit() for c in etiqueta) for etiqueta in d.tags)
