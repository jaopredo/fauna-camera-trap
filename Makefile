# Alvos de uso diario. Tudo roda no .venv criado por `make setup`: o `python3` do
# sistema nao tem as dependencias do projeto.
PY := .venv/bin/python

# ===================== Ambiente =====================
# torch e torchvision sao instalados a parte, nunca pelo pyproject: a roda certa
# depende da placa (CUDA, ROCm ou CPU) e fixar isso nas dependencias obrigaria
# todo mundo ao mesmo backend. `--torch-backend=auto` detecta o hardware — numa
# Radeon ele resolve para `+rocmX.Y`, numa NVIDIA para `+cuXXX`.
setup:
	uv sync
	uv pip install torch torchvision --torch-backend=auto

# Forca a roda de CPU: Colab sem GPU, CI, ou para conferir o tempo de inferencia
# do modelo entregue, que o enunciado exige em CPU.
setup-cpu:
	uv sync
	uv pip install torch torchvision --torch-backend=cpu

# Regera requirements.txt a partir do pyproject/uv.lock. Precisa de uv; e so para
# manutencao do arquivo versionado, nao faz parte do fluxo de instalacao sem uv.
# torch nao entra nele, pelo mesmo motivo de nao estar no pyproject.
requirements.txt: pyproject.toml
	uv export --no-hashes --no-dev --format requirements.txt -o requirements.txt

# Equivalente a `make setup`, mas sem uv: venv + pip e o requirements.txt versionado.
setup-pip:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install torch torchvision

# Equivalente a `make setup-cpu`, mas sem uv.
setup-cpu-pip:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Mostra qual roda de torch esta instalada e se a GPU foi encontrada.
ambiente:
	@$(PY) -c "import torch, torchvision; \
print('torch', torch.__version__, '| torchvision', torchvision.__version__); \
print('GPU disponivel:', torch.cuda.is_available(), \
'|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'rodando em CPU')"

# ===================== Qualidade =====================
test:
	$(PY) -m pytest -q

fmt:
	uvx ruff@0.14.3 format src tests scripts predict.py
	uvx ruff@0.14.3 check src tests scripts predict.py

# ===================== Parte 0 — o piso =====================
# Quantas seeds cada experimento roda esta declarado no proprio YAML
# (`seeds: [0, 42, 123]`), nao aqui: o protocolo e configuracao, nao shell.
parte0:
	$(PY) scripts/train.py --config configs/p0_scratch.yaml
	$(PY) scripts/train.py --config configs/p0_finetune.yaml
	$(MAKE) parte0-relatorio

# So o relatorio, a partir das execucoes que ja existem em outputs/.
parte0-relatorio:
	$(PY) scripts/report_p0.py

# Acompanha as execucoes em andamento sem interromper nada.
progresso:
	@$(PY) scripts/progresso.py

.PHONY: setup setup-cpu setup-pip setup-cpu-pip ambiente test fmt \
        parte0 parte0-relatorio progresso
