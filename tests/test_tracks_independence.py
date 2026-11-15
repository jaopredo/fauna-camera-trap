"""As trilhas sao independentes — assercao, nao promessa.

O requisito: apagar uma trilha nao pode afetar a outra de nenhuma forma. Isso tem
duas partes, e as duas sao testadas aqui:

1. **Nenhuma trilha importa de outra.** Se ``src/tracks/c/`` importasse algo de
   ``src/tracks/b/``, apagar B quebraria C. O teste percorre a arvore sintatica de
   cada modulo de cada trilha e falha se encontrar esse import.
2. **Nada fora de ``src/tracks/`` importa uma trilha especifica.** A infraestrutura
   comum e os scripts neutros (``report_p0``, ``report_p2``, ``train``,
   ``predict.py``) nao podem depender de uma trilha, senao apagar a trilha quebraria
   o projeto. Os relatorios **daquela** trilha podem — eles sao parte dela e sao
   apagados junto.

Analise estatica com ``ast``, nao ``import``: um import de verdade so pegaria o que
executa, e um import dentro de funcao passaria batido.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.tracks import AVAILABLE_TRACKS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKS_ROOT = PROJECT_ROOT / "src" / "tracks"

#: Scripts que pertencem a uma trilha e sao apagados junto com ela. A chave e o
#: caminho; o valor, a trilha de que ele faz parte.
SCRIPTS_DE_TRILHA = {
    "scripts/report_p1.py": "b",
    "scripts/report_p1c.py": "c",
    "scripts/teacher.py": "c",
}


def modulos_de(raiz: Path) -> list[Path]:
    return sorted(p for p in raiz.rglob("*.py") if "__pycache__" not in p.parts)


def imports_de(caminho: Path) -> set[str]:
    """Todos os modulos que este arquivo importa, inclusive dentro de funcoes."""
    arvore = ast.parse(caminho.read_text(encoding="utf-8"), filename=str(caminho))
    nomes: set[str] = set()
    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            nomes.update(alias.name for alias in no.names)
        elif isinstance(no, ast.ImportFrom) and no.module and no.level == 0:
            nomes.add(no.module)
            # `from src.tracks import b` tambem e um import da trilha b
            nomes.update(f"{no.module}.{alias.name}" for alias in no.names)
    return nomes


def trilhas_citadas(nomes: set[str], exceto: str | None = None) -> set[str]:
    """Quais trilhas aparecem nestes imports, fora a propria."""
    encontradas = set()
    for nome in nomes:
        for trilha in AVAILABLE_TRACKS:
            if nome == f"src.tracks.{trilha}" or nome.startswith(
                f"src.tracks.{trilha}."
            ):
                if trilha != exceto:
                    encontradas.add(trilha)
    return encontradas


def test_ha_pelo_menos_uma_trilha():
    """Guarda contra suite vazia: sem trilha, os testes abaixo nao provariam nada.

    Nao e acoplamento. Com duas trilhas instaladas, apagar uma deixa a outra e esta
    guarda continua passando. Ela so acusa o caso degenerado de zero trilhas, em que
    os testes de independencia passariam sem olhar arquivo nenhum.
    """
    assert AVAILABLE_TRACKS, (
        "nenhuma trilha em src/tracks/ — os testes de independencia ficariam vazios. "
        "O resto da suite deve continuar passando neste estado: a infraestrutura nao "
        "depende de trilha nenhuma."
    )


@pytest.mark.parametrize("trilha", AVAILABLE_TRACKS)
def test_trilha_nao_importa_outra_trilha(trilha):
    """Se B importasse de C, apagar C quebraria B."""
    for modulo in modulos_de(TRACKS_ROOT / trilha):
        invasoras = trilhas_citadas(imports_de(modulo), exceto=trilha)
        assert not invasoras, (
            f"{modulo.relative_to(PROJECT_ROOT)} (trilha {trilha}) importa da "
            f"trilha {sorted(invasoras)}; apagar aquela trilha quebraria esta"
        )


def test_infraestrutura_nao_importa_trilha_nenhuma():
    """`src/` fora de `src/tracks/` tem de rodar sem trilha alguma instalada.

    O unico import permitido e o de `src/tracks/__init__.py`, que e a descoberta.
    """
    raiz = PROJECT_ROOT / "src"
    for modulo in modulos_de(raiz):
        if TRACKS_ROOT in modulo.parents or modulo == TRACKS_ROOT / "__init__.py":
            continue
        citadas = trilhas_citadas(imports_de(modulo))
        assert not citadas, (
            f"{modulo.relative_to(PROJECT_ROOT)} e infraestrutura comum e importa a "
            f"trilha {sorted(citadas)}; apagar a trilha quebraria o projeto"
        )


def test_scripts_neutros_nao_importam_trilha_nenhuma():
    """`train.py`, `report_p0.py`, `report_p2.py` e `predict.py` valem sem trilha."""
    caminhos = [PROJECT_ROOT / "predict.py"] + modulos_de(PROJECT_ROOT / "scripts")
    for caminho in caminhos:
        relativo = caminho.relative_to(PROJECT_ROOT).as_posix()
        dono = SCRIPTS_DE_TRILHA.get(relativo)
        citadas = trilhas_citadas(imports_de(caminho), exceto=dono)
        assert not citadas, (
            f"{relativo} nao pertence a trilha {sorted(citadas)} mas importa dela; "
            f"declare-o em SCRIPTS_DE_TRILHA ou remova o import"
        )


@pytest.mark.parametrize("trilha", AVAILABLE_TRACKS)
def test_trilha_se_registra_ao_ser_descoberta(trilha):
    """A descoberta tem de ter efeito: um `__init__.py` vazio passaria calado."""
    from src.data.augment import available_augmentations
    from src.methods import available_strategies

    registrado = set(available_strategies()) | set(available_augmentations())
    pacote = __import__(f"src.tracks.{trilha}", fromlist=["*"])
    assert pacote.__all__, f"src/tracks/{trilha}/__init__.py nao exporta nada"
    assert registrado, "nenhuma estrategia ou aumentacao registrada"
