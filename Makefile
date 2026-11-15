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

# Dependencia so da trilha C: o professor zero-shot (BioCLIP, CLIP). Fica fora do
# pyproject pelo mesmo motivo do torch (ver D11): o `uv sync` reconcilia o ambiente
# com o lock, e um `open_clip_torch` declarado la arrastaria torch do indice padrao
# da PyPI por cima da roda ROCm. Instalado DEPOIS do torch, o uv o ve satisfeito e
# nao o toca — verificado com `uv pip install --dry-run`.
#
# Nao entra no `make setup` de proposito: quem nao roda a trilha C nao precisa
# baixar ~600 MB de pesos, e apagar a trilha nao deixa dependencia orfa.
#
# ATENCAO ao citar comandos dentro de `python -c` numa receita: com aspas DUPLAS por
# fora, o shell interpreta crase como substituicao de comando. Uma mensagem com
# `make setup` entre crases chegou a EXECUTAR `make setup` — que roda `uv sync` e
# desinstala justamente o open_clip que a linha acima instalou. Por isso as aspas de
# fora sao SIMPLES aqui e no alvo `ambiente`: dentro delas, crase e $ sao literais.
setup-professor:
	uv pip install open_clip_torch
	@$(PY) -c 'import torch, open_clip; \
print("open_clip", open_clip.__version__, "| torch", torch.__version__); \
print("a roda de torch sobreviveu" if "rocm" in torch.__version__ or "cu" in torch.__version__ \
else "ATENCAO: torch sem backend de GPU - rode make setup de novo")'

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
	@$(PY) -c 'import torch, torchvision; \
print("torch", torch.__version__, "| torchvision", torchvision.__version__); \
print("GPU disponivel:", torch.cuda.is_available(), \
"|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "rodando em CPU")'

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

# ===================== Parte 1 — Trilha B (FixMatch) =====================
# 1) Controle do confundidor: o piso com a mesma aumentacao fraca do FixMatch.
#    Sem ele, parte de qualquer ganho poderia vir da aumentacao, nao dos dados
#    sem rotulo.
parte1-controle:
	$(PY) scripts/train.py --config configs/p0_finetune_weak.yaml

# 2) Varredura de tau, uma seed cada. O `name` muda junto porque o diretorio da
#    execucao e <nome>_b<orcamento>_s<seed> — sem isso as duas se sobrescreveriam.
parte1-sweep:
	$(PY) scripts/train.py --config configs/p1_fixmatch.yaml \
	    --set name=p1_tau070 strategy.params.tau=0.70 seeds=[0]
	$(PY) scripts/train.py --config configs/p1_fixmatch.yaml \
	    --set name=p1_tau080 strategy.params.tau=0.80 seeds=[0]
	$(PY) scripts/report_p1.py --sweep p1_tau

# 3) Final: as duas variantes, 3 seeds cada. FixMatch puro mostra o vies;
#    FixMatch+DA testa a correcao.
parte1:
	$(PY) scripts/train.py --config configs/p1_fixmatch.yaml
	$(PY) scripts/train.py --config configs/p1_fixmatch_da.yaml
	$(MAKE) parte1-relatorio

parte1-relatorio:
	$(PY) scripts/report_p1.py

# ===================== Parte 1 — Trilha C (professor e aluno) =============
# 1) O professor rotula. Roda UMA vez e grava .npz; depois disso ele sai de cena, e
#    e por isso que "o professor nao pode fazer parte do modelo entregue" e
#    estrutural e nao disciplina. Precisa de `make setup-professor`.
#    O val nao entra: o script o recusa com excecao.
parte1c-professor:
	$(PY) scripts/teacher.py

# 2) Os tres tratamentos de ruido do obrigatorio 3, uma seed cada:
#    sem tratamento / filtro por confianca / rotulos suaves com temperatura.
#    O filtro e a escolha ingenua; a trilha B preve que ela colapsa classes (D17).
parte1c-sweep:
	$(PY) scripts/train.py --config configs/p1c_distill_hard.yaml   --set seeds=[0]
	$(PY) scripts/train.py --config configs/p1c_distill_filter.yaml --set seeds=[0]
	$(PY) scripts/train.py --config configs/p1c_distill_soft.yaml   --set seeds=[0]
	$(MAKE) parte1c-relatorio

# 3) Tudo: professor, varredura e relatorio.
parte1c:
	$(MAKE) parte1c-professor
	$(MAKE) parte1c-sweep

parte1c-relatorio:
	$(PY) scripts/report_p1c.py

# ===================== Parte 2 — a curva de orcamento =====================
# Os orcamentos de 1 e 20, uma seed cada; o de 5 ja saiu das Partes 0 e 1. Nao ha
# config novo: o orcamento e campo de configuracao e o diretorio da execucao ja
# inclui b<orcamento>, entao nada se sobrescreve.
#
# Custo: o piso roda em ~1 min, o FixMatch em ~20 min. A epoca do FixMatch e
# definida pelo pool sem rotulo (6.505 imagens), que nao encolhe com o orcamento —
# baixar o orcamento nao deixa a execucao mais rapida.
# O metodo da trilha e o aluno da Trilha C com filtro por confianca — o que ganhou
# no val (D20). A Trilha B ficou como registro de aprendizado e nao entra na curva;
# se a apresentacao quiser o ponto dela, ha `parte2-fixmatch` abaixo.
#
# Custo: o piso roda em ~1 min por execucao, o aluno em ~11 min (22 s/epoca x 30).
# A epoca do aluno e definida pelo pool de 6.505, que nao encolhe com o orcamento —
# baixar o orcamento nao deixa a execucao mais rapida.
parte2:
	$(PY) scripts/train.py --config configs/p0_finetune.yaml --set data.budget=1 seeds=[0]
	$(PY) scripts/train.py --config configs/p0_finetune.yaml --set data.budget=20 seeds=[0]
	$(PY) scripts/train.py --config configs/p1c_distill_filter.yaml --set data.budget=1 seeds=[0]
	$(PY) scripts/train.py --config configs/p1c_distill_filter.yaml --set data.budget=20 seeds=[0]
	$(MAKE) parte2-orcamentos
	$(MAKE) parte2-relatorio

# Opcional: os pontos de 1 e 20 do FixMatch, se a apresentacao quiser as duas trilhas
# na mesma curva. ~40 min, e nao e obrigatorio — o enunciado pede o metodo de UMA
# trilha, e a nossa e a C.
parte2-fixmatch:
	$(PY) scripts/train.py --config configs/p1_fixmatch.yaml --set data.budget=1 seeds=[0]
	$(PY) scripts/train.py --config configs/p1_fixmatch.yaml --set data.budget=20 seeds=[0]
	$(PY) scripts/report_p1.py --orcamento 1
	$(PY) scripts/report_p1.py --orcamento 20

# Avalia no test_public as execucoes de 1 e 20 e acrescenta as linhas ao
# curva_orcamento.csv. Os relatorios das Partes 0 e 1 reportam so o orcamento de 5
# (e filtram por ele), entao cada orcamento entra no CSV por uma chamada propria.
parte2-orcamentos:
	$(PY) scripts/report_p0.py --orcamento 1
	$(PY) scripts/report_p0.py --orcamento 20
	$(PY) scripts/report_p1c.py --orcamento 1
	$(PY) scripts/report_p1c.py --orcamento 20

# A figura obrigatoria e a tabela. Le so o curva_orcamento.csv: sem GPU, sem modelo.
parte2-relatorio:
	$(PY) scripts/report_p2.py

# ===================== Parte 3 — galeria de falhas e atalhos ==============
# Nao treina nada. Carrega o best.pt da execucao de melhor F1 de VALIDACAO no
# orcamento (a mesma regra que define o modelo entregue) e faz um forward pelo
# test_public para medir o mapa CAM de todas as 811 imagens, mais 8 imagens com
# mapa, oclusao e diagnostico. ~1 min em GPU, ~4 min em CPU.
#
# Nenhum nome de experimento esta escrito no script: ele procura todas as execucoes
# do orcamento e ordena pelo val. Trocar o modelo entregue nao exige editar nada.
parte3:
	$(PY) scripts/report_p3.py

# Em CPU, para rodar enquanto um treino ocupa a GPU (ver docs/STATUS.md: nunca dois
# trabalhos de GPU ao mesmo tempo nesta placa).
parte3-cpu:
	$(PY) scripts/report_p3.py --device cpu

# Acompanha as execucoes em andamento sem interromper nada.
progresso:
	@$(PY) scripts/progresso.py

.PHONY: setup setup-cpu setup-pip setup-cpu-pip setup-professor ambiente test fmt \
        parte0 parte0-relatorio parte1 parte1-controle parte1-sweep \
        parte1-relatorio parte1c parte1c-professor parte1c-sweep \
        parte1c-relatorio parte2 parte2-fixmatch parte2-orcamentos \
        parte2-relatorio parte3 parte3-cpu progresso
