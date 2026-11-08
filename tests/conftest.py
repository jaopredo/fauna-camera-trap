"""Fixtures comuns dos testes."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_root() -> Path:
    """Raiz do pacote de dados.

    O pacote fica fora do Git (415 MB), entao os testes que dependem dele sao
    marcados como pulados numa maquina limpa, em vez de falharem.
    """
    if not (DATA_ROOT / "classes.csv").exists():
        pytest.skip(f"pacote de dados ausente em {DATA_ROOT}")
    return DATA_ROOT
