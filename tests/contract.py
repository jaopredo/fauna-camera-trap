"""O contrato comum das Strategies, num helper reusavel.

Nao e um arquivo de teste (nao casa com ``test_*``): e a assercao que
``tests/test_strategy_contract.py`` roda sobre as estrategias construiveis sem
artefato externo, e que o teste **de cada trilha** roda sobre a estrategia dela,
montada com o que ela precisa.

Assim o contrato e um so, verificado em todas as estrategias, sem que o teste
compartilhado precise saber o que uma trilha especifica exige para nascer — nem
carregar uma lista de excecoes que ficaria obsoleta ao apagar uma trilha.
"""

from __future__ import annotations

from src.data.loaders import LABELED, UNLABELED
from src.methods import TrainingStrategy

#: Fontes de dados que o laco de treino sabe montar (``src.data.loaders``).
FONTES_CONHECIDAS = {LABELED, UNLABELED}


def assert_cumpre_o_contrato(strategy: object, nome: str) -> None:
    """O L do SOLID: toda estrategia e substituivel no laco de treino."""
    assert isinstance(strategy, TrainingStrategy), (
        f"{nome}: {type(strategy).__name__} nao e um TrainingStrategy"
    )
    assert strategy.required_sources, f"{nome}: nao declarou required_sources"
    assert strategy.primary_source in strategy.required_sources, (
        f"{nome}: primary_source {strategy.primary_source!r} nao esta em "
        f"required_sources {strategy.required_sources}"
    )
    desconhecidas = set(strategy.required_sources) - FONTES_CONHECIDAS
    assert not desconhecidas, (
        f"{nome}: pede fontes que o laco nao sabe montar: {sorted(desconhecidas)}"
    )
    assert isinstance(strategy.state_dict(), dict), (
        f"{nome}: state_dict nao devolveu dict; o checkpoint nao saberia guardar"
    )
