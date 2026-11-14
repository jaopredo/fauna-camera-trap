"""Carregamento e validacao da configuracao."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ConfigError, ExperimentConfig, load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

#: Configuracao minima valida. ``model`` e ``strategy`` sao obrigatorios: uma
#: execucao sem backbone e sem metodo declarados nao tem o que rodar.
BASE = """
name: minimo
seeds: [0]
model:
  name: backbone_qualquer
strategy:
  name: metodo_qualquer
"""


def escrever(tmp_path: Path, texto: str, nome: str = "config.yaml") -> Path:
    path = tmp_path / nome
    path.write_text(texto, encoding="utf-8")
    return path


def test_exemplo_do_repositorio_carrega():
    config = load_config(CONFIGS / "exemplo.yaml")
    assert config.name == "exemplo"
    assert config.data.budget == 5
    assert config.optim.monitor == "macro_f1"


def test_secoes_omitidas_usam_os_padroes(tmp_path):
    config = load_config(escrever(tmp_path, BASE))
    assert config.data.budget == 5
    assert config.data.image_size == 224
    assert config.optim.epochs == 30
    assert config.output_dir == Path("outputs")


def test_configuracao_e_imutavel():
    config = load_config(CONFIGS / "exemplo.yaml")
    with pytest.raises(Exception):
        config.seeds = (7,)  # type: ignore[misc]


def test_carregar_duas_vezes_devolve_objetos_independentes(tmp_path):
    """Sem Singleton: a Parte 2 carrega varias configuracoes no mesmo processo.

    Com uma instancia global compartilhada, as 3 seeds do orcamento de 5 rodariam
    todas com a primeira configuracao, sem nada no log indicando o problema.
    """
    primeiro = escrever(tmp_path, BASE.replace("name: minimo", "name: a"), "a.yaml")
    segundo = escrever(
        tmp_path,
        BASE.replace("name: minimo", "name: b").replace("seeds: [0]", "seeds: [1]"),
        "b.yaml",
    )

    a = load_config(primeiro)
    b = load_config(segundo)
    assert (a.name, a.seeds) == ("a", (0,))
    assert (b.name, b.seeds) == ("b", (1,))
    assert a is not b


def test_chave_desconhecida_na_raiz_e_recusada(tmp_path):
    path = escrever(tmp_path, BASE + "sede: 3\n")
    with pytest.raises(ConfigError, match="raiz: chaves desconhecidas"):
        load_config(path)


def test_chave_desconhecida_em_secao_e_recusada(tmp_path):
    # typo que passaria em silencio e faria o treino rodar com outro lr
    path = escrever(tmp_path, BASE + "optim:\n  Ir: 0.1\n")
    with pytest.raises(ConfigError, match="optim: chaves desconhecidas"):
        load_config(path)


def test_model_e_obrigatorio(tmp_path):
    path = escrever(tmp_path, "name: a\nstrategy:\n  name: m\n")
    with pytest.raises(ConfigError, match="model: chaves obrigatorias ausentes"):
        load_config(path)


def test_strategy_e_obrigatorio(tmp_path):
    path = escrever(tmp_path, "name: a\nmodel:\n  name: b\n")
    with pytest.raises(ConfigError, match="strategy: chaves obrigatorias ausentes"):
        load_config(path)


def test_name_e_obrigatorio(tmp_path):
    path = escrever(tmp_path, BASE.replace("name: minimo\n", ""))
    with pytest.raises(ConfigError, match="name"):
        load_config(path)


def test_orcamento_invalido_e_recusado(tmp_path):
    from src.data.splits import InvalidBudgetError

    path = escrever(tmp_path, BASE + "data:\n  budget: 7\n")
    with pytest.raises(InvalidBudgetError):
        load_config(path)


@pytest.mark.parametrize(
    "secao,chave,valor",
    [
        ("data", "image_size", 0),
        ("data", "batch_size", -1),
        ("data", "num_workers", -1),
        ("optim", "epochs", 0),
        ("optim", "lr", 0),
        ("optim", "save_every", 0),
    ],
)
def test_valores_sem_sentido_sao_recusados(tmp_path, secao, chave, valor):
    path = escrever(tmp_path, BASE + f"{secao}:\n  {chave}: {valor}\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_sobrescritas_convertem_tipo_como_o_yaml(tmp_path):
    """A Parte 2 varre seeds e orcamentos por --set, sem um YAML por combinacao."""
    path = escrever(tmp_path, BASE)
    config = load_config(
        path, overrides=["seeds=[3,4]", "data.budget=20", "model.pretrained=true"]
    )
    assert config.seeds == (3, 4)
    assert config.data.budget == 20
    assert config.model.pretrained is True


def test_sobrescrita_sem_igual_e_recusada(tmp_path):
    with pytest.raises(ConfigError, match="chave=valor"):
        load_config(escrever(tmp_path, BASE), overrides=["seeds"])


def test_sobrescrita_de_chave_desconhecida_e_recusada(tmp_path):
    with pytest.raises(ConfigError, match="desconhecidas"):
        load_config(escrever(tmp_path, BASE), overrides=["optim.momento=0.9"])


def test_as_dict_e_serializavel():
    import json

    config = load_config(CONFIGS / "exemplo.yaml")
    dados = json.loads(json.dumps(config.as_dict()))
    assert dados["data"]["root"] == "data"
    assert dados["model"]["name"] == "TBD"


def test_configuracao_ausente_tem_erro_claro(tmp_path):
    with pytest.raises(ConfigError, match="nao encontrada"):
        load_config(tmp_path / "nao_existe.yaml")


def test_from_raw_aceita_dicionario_direto():
    config = ExperimentConfig.from_raw(
        {
            "name": "x",
            "model": {"name": "b"},
            "strategy": {"name": "m"},
            "optim": {"epochs": 2},
        }
    )
    assert config.optim.epochs == 2
    assert config.model.name == "b"


# ===================== protocolo de seeds no config =========================
# O enunciado pede 3 seeds no orcamento de 5 e uma nos de 1 e 20. Essa regra mora
# no YAML do experimento, nao num laco de shell — estes testes guardam o contrato.


def test_experimento_declara_o_conjunto_de_seeds(tmp_path):
    config = load_config(
        escrever(tmp_path, BASE.replace("seeds: [0]", "seeds: [0, 42, 123]"))
    )
    assert config.seeds == (0, 42, 123)


def test_ordem_declarada_e_preservada(tmp_path):
    """A primeira seed e a que gera as figuras; ordenar mudaria qual é."""
    config = load_config(
        escrever(tmp_path, BASE.replace("seeds: [0]", "seeds: [42, 0, 7]"))
    )
    assert config.seeds == (42, 0, 7)


def test_inteiro_solto_vira_uma_seed(tmp_path):
    config = load_config(escrever(tmp_path, BASE.replace("seeds: [0]", "seeds: 7")))
    assert config.seeds == (7,)


def test_seeds_vazia_e_recusada(tmp_path):
    with pytest.raises(ConfigError, match="ao menos uma seed"):
        load_config(escrever(tmp_path, BASE.replace("seeds: [0]", "seeds: []")))


def test_seeds_repetidas_sao_recusadas(tmp_path):
    """Repetir sobrescreveria o mesmo diretorio e falsearia a media +- desvio."""
    with pytest.raises(ConfigError, match="repetidas"):
        load_config(escrever(tmp_path, BASE.replace("seeds: [0]", "seeds: [0, 42, 0]")))


def test_seeds_nao_numericas_sao_recusadas(tmp_path):
    with pytest.raises(ConfigError, match="inteiro"):
        load_config(escrever(tmp_path, BASE.replace("seeds: [0]", "seeds: [a, b]")))


def test_campo_seed_antigo_aponta_para_o_novo(tmp_path):
    """Config escrito antes da mudanca falha explicando o que fazer."""
    with pytest.raises(ConfigError, match=r"seeds: \[0, 42, 123\]"):
        load_config(escrever(tmp_path, BASE.replace("seeds: [0]", "seed: 0")))


def test_nome_da_execucao_vem_do_config(tmp_path):
    """Fonte unica da convencao: o treino grava com ela, o relatorio procura por ela."""
    config = load_config(
        escrever(
            tmp_path,
            BASE.replace("seeds: [0]", "seeds: [0, 42]").replace(
                "name: minimo", "name: p0_finetune"
            ),
        )
    )
    assert config.run_name(42) == "p0_finetune_b5_s42"


def test_nome_da_execucao_recusa_seed_nao_declarada(tmp_path):
    config = load_config(escrever(tmp_path, BASE))
    with pytest.raises(ConfigError, match="nao esta declarada"):
        config.run_name(99)


def test_manifesto_registra_o_conjunto_declarado(tmp_path):
    config = load_config(
        escrever(tmp_path, BASE.replace("seeds: [0]", "seeds: [0, 42]"))
    )
    assert config.as_dict()["seeds"] == [0, 42]
