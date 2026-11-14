"""Interface de entrega: ``python predict.py --imagens PASTA --saida preds.csv``.

Roda o script de verdade, em subprocesso, como o monitor vai rodar. O que se
verifica e o contrato do enunciado: colunas ``image_id,label``, nomes vindos de
``classes.csv``, contagem de parametros impressa e nenhum download de pesos.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.config.schema import ModelConfig
from src.data.catalog import ClassVocabulary
from src.data.splits import Split
from src.models import build_model, save_checkpoint
from src.models.checkpoint import CheckpointState

PROJECT_ROOT = Path(__file__).resolve().parent.parent
N_IMAGENS = 4

CONFIG = """
name: teste_entrega
seeds: [0]
data:
  root: {root}
  image_size: 64
model:
  name: small_cnn
  pretrained: false
strategy:
  name: supervised
"""


@pytest.fixture
def entrega(tmp_path, data_root):
    """Monta pesos, config e uma pasta de imagens, como numa entrega real."""
    classes = ClassVocabulary.from_csv(data_root / "classes.csv")
    modelo = build_model(ModelConfig(name="small_cnn"), num_classes=len(classes))
    pesos = save_checkpoint(
        tmp_path / "modelo.pt",
        model=modelo,
        state=CheckpointState(epoch=1, best_metric=0.0, monitor="macro_f1"),
    )
    config = tmp_path / "entrega.yaml"
    config.write_text(CONFIG.format(root=data_root), encoding="utf-8")

    imagens = tmp_path / "imagens"
    imagens.mkdir()
    origem = data_root / "images" / Split.VAL.value
    for caminho in sorted(origem.glob("*.jpg"))[:N_IMAGENS]:
        shutil.copy(caminho, imagens / caminho.name)

    return {
        "pesos": pesos,
        "config": config,
        "imagens": imagens,
        "saida": tmp_path / "preds.csv",
        "classes": data_root / "classes.csv",
        "vocabulario": classes,
    }


def rodar(entrega) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--imagens",
            str(entrega["imagens"]),
            "--saida",
            str(entrega["saida"]),
            "--config",
            str(entrega["config"]),
            "--pesos",
            str(entrega["pesos"]),
            "--classes",
            str(entrega["classes"]),
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=300,
    )


def test_gera_csv_com_as_colunas_do_enunciado(entrega):
    processo = rodar(entrega)
    assert processo.returncode == 0, processo.stderr

    with entrega["saida"].open(newline="", encoding="utf-8") as handle:
        leitor = csv.reader(handle)
        assert next(leitor) == ["image_id", "label"]
        linhas = list(leitor)

    assert len(linhas) == N_IMAGENS
    esperados = {p.stem for p in entrega["imagens"].glob("*.jpg")}
    assert {linha[0] for linha in linhas} == esperados


def test_rotulos_saem_de_classes_csv(entrega):
    """O enunciado exige os nomes de classes.csv, nao indices nem nomes inventados."""
    rodar(entrega)
    with entrega["saida"].open(newline="", encoding="utf-8") as handle:
        rotulos = {linha["label"] for linha in csv.DictReader(handle)}
    assert rotulos <= set(entrega["vocabulario"].labels)


def test_imprime_a_contagem_de_parametros(entrega):
    """Exigencia explicita da tabela de entregaveis."""
    processo = rodar(entrega)
    assert "parametros:" in processo.stdout
    assert "394,293" in processo.stdout


def test_nao_baixa_pesos(entrega, monkeypatch):
    """Sem internet na inferencia: os pesos vem do checkpoint local.

    Se o predict.py construisse o modelo com pretrained=True, o torch tentaria
    baixar para o cache. Apontar o cache para um diretorio vazio e somente-leitura
    faria essa tentativa falhar; aqui basta conferir que nada foi escrito nele.
    """
    cache = entrega["saida"].parent / "cache_torch"
    cache.mkdir()
    ambiente = {**dict(__import__("os").environ), "TORCH_HOME": str(cache)}
    processo = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--imagens",
            str(entrega["imagens"]),
            "--saida",
            str(entrega["saida"]),
            "--config",
            str(entrega["config"]),
            "--pesos",
            str(entrega["pesos"]),
            "--classes",
            str(entrega["classes"]),
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=ambiente,
        timeout=300,
    )
    assert processo.returncode == 0, processo.stderr
    assert not list(cache.rglob("*.pth")), "houve download de pesos na inferencia"


def test_pasta_vazia_falha_com_mensagem_clara(entrega, tmp_path):
    vazia = tmp_path / "vazia"
    vazia.mkdir()
    processo = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--imagens",
            str(vazia),
            "--saida",
            str(entrega["saida"]),
            "--config",
            str(entrega["config"]),
            "--pesos",
            str(entrega["pesos"]),
            "--classes",
            str(entrega["classes"]),
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=300,
    )
    assert processo.returncode != 0
    assert "nenhuma imagem" in processo.stdout + processo.stderr
