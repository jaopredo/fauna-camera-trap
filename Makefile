# Alvos de uso diario. `uv` e opcional: `setup-pip` faz o mesmo com venv + pip.

setup:
	uv sync

# Instalacao sem uv, a partir do requirements.txt versionado.
setup-pip:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

# Regera requirements.txt a partir do pyproject/uv.lock. So manutencao.
requirements.txt: pyproject.toml
	uv export --no-hashes --no-dev --format requirements.txt -o requirements.txt

test:
	python3 -m pytest -q

fmt:
	uvx ruff@0.14.3 format src tests scripts predict.py
	uvx ruff@0.14.3 check src tests scripts predict.py

.PHONY: setup setup-pip test fmt
