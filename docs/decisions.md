# Decisões

Registro das decisões de arquitetura e de modelagem, com alternativas descartadas.
Decisão de modelagem cita a aula consultada; decisão de arquitetura cita o
enunciado e o que varia de fato.

---

## D1 — Organização do pacote por eixo de variação (08/11)

**Decisão.** `src/` dividido em `config`, `data`, `models`, `methods`,
`training`, `evaluation`, `analysis`, mais os módulos folha `metrics.py`,
`tracking.py` e `errors.py`. Composição e injeção só em `scripts/` e `predict.py`.

**Responsabilidade.** Cada pacote tem um único motivo para mudar: `data` muda se o
pacote de dados mudar; `methods`, se a trilha mudar; `metrics`, se a métrica mudar.

**Variação real (do enunciado, não especulada).**

| Eixo | De onde vem |
|---|---|
| Backbone | Parte 0 exige 2 (CNN do zero e ImageNet); Trilha A exige 3 lado a lado |
| Método de treino | Parte 0 supervisionado + o método da trilha escolhida |
| Aumentação | experimento obrigatório da Trilha A; par fraco/forte do FixMatch |
| Orçamento × seed | 1, 5 e 20 rótulos; 3 seeds no orçamento de 5 (Parte 2) |
| Conjunto de avaliação | `val` para selecionar, `test_public` para reportar |

**Padrões e os nomes que eles dão às classes.** Factory com Registry em `models`
e `methods`; Strategy em
`data/augment.py` e em `methods`; Observer em `training/observers.py`. Sem padrão
em `metrics`, `config` e `tracking`: função pura e dataclass congelada.

Onde e como cada um foi implementado, com os pontos de extensão: seção
**Padrões de projeto** do `README.md`.

**Alternativas descartadas.**

1. *Espelhar `deep-learning/semantic-segmentation`*, com engines paralelos para
   treino, validação e teste. Descartada porque nesse projeto
   `src/training/config.py` e `src/evaluation/config.py` são arquivos idênticos —
   o sintoma de dividir por *quando se roda* em vez de por *o que varia*.
2. *Módulos planos, sem registries, até existir a segunda implementação.* Mais
   rápido para o M0, mas a segunda implementação de backbone e de método já é
   exigida pelo enunciado, então a refatoração seria certa no M2.

---

## D2 — Validação e teste no mesmo módulo, com barreira tipada (08/11)

**Decisão.** Uma função `evaluate(model, loader, split=...)` serve `val` e
`test_public`. O `EvaluationResult` carrega o próprio `Split`, e toda função de
`src/evaluation/selection.py` chama `ensure_selectable`, que recusa qualquer
conjunto fora de `{val}` com `SplitPermissionError`.

**Por quê.** Validação e teste fazem a mesma computação — mesmo forward, mesmas
métricas. O que difere é permissão: o enunciado (seção 3) proíbe treinar ou
selecionar modelo no `test_public`. Como o resultado sabe de onde veio, não existe
caminho de seleção que aceite número de teste sem levantar exceção, e isso não
depende de quem chama lembrar da regra.

**Coberto por** `tests/test_selection.py`, que inclui um teste de contrato: ele
enumera as funções públicas de `selection.py` por introspecção e falha se alguma
função nova não estiver coberta pela barreira.

**Alternativa descartada.** Um módulo `test/` paralelo ao de validação: duplicaria
a computação e não garantiria nada — a regra continuaria dependendo de disciplina
de quem chama. Além disso o nome colidiria com `tests/` do pytest.

---

## D3 — Configuração sem Singleton (08/11)

**Decisão.** `load_config(path, overrides=...)` devolve uma `ExperimentConfig`
congelada nova a cada chamada. Seções recusam chave desconhecida; `model` e
`strategy` são obrigatórios.

**Por quê.** O projeto de referência usa `AppConfig` com metaclass Singleton: a
segunda chamada com outro YAML devolve silenciosamente a configuração da primeira.
A Parte 2 roda 3 seeds × 3 orçamentos; num runner de processo único, as três seeds
rodariam iguais e nada no log indicaria o problema. `CLAUDE.md` também proíbe
estado global.

**Coberto por** `tests/test_config.py::test_carregar_duas_vezes_devolve_objetos_independentes`.

---

## D4 — F1 macro sempre sobre as 21 classes (08/11)

**Decisão.** `src/metrics.py` calcula tudo a partir da matriz de confusão
`21 × 21`; classe sem suporte e sem predição entra na média macro como F1 zero.
Equivale a `sklearn.metrics.f1_score(labels=range(21), zero_division=0)`, mas sem
depender do sklearn no módulo folha.

**Por quê.** O `val` tem 202 imagens e 2 de cavalo; o `test_public` tem 4. Uma
média sobre "as classes observadas" dividiria por 20 em vez de 21 e devolveria um
número inflado, incomparável com os demais. O enunciado é explícito: a métrica
principal pesa cavalo igual a urso.

**Coberto por** `tests/test_metrics.py::test_classe_ausente_entra_na_media_como_zero`.

---

## D5 — Python 3.12 (08/11)

**Decisão.** `requires-python = ">=3.12,<3.13"`.

**Por quê.** O `CLAUDE.md` exige rodar no Colab e localmente, e nada no projeto
precisa de sintaxe mais nova. Em 3.14 as rodas existem mas ficam atrás: a versão
mais recente de `torchvision` no PyPI (0.29.2) não publica roda `cp314`, só até a
0.29.1.

---

## D7 — Nomes das classes seguem o padrão que implementam (08/11)

**Decisão.** Cada participante de um padrão leva o nome do papel que exerce:

| Padrão | Classe / função |
|---|---|
| Strategy (treino) | `TrainingStrategy`, `register_strategy`, `build_strategy` |
| Strategy (aumentação) | `AugmentationStrategy`, `register_augmentation` |
| Factory com Registry | `ModelFactory`, `register_model`, `build_model` |
| Observer | `TrainingObserver` e os concretos `ConsoleLogger`, `HistoryWriter`, `CheckpointWriter` |

**Por quê.** O nome do papel é o que diz onde uma implementação nova entra. Quem lê
`TrainingStrategy` já sabe que FixMatch é um irmão dela, não uma edição do laço.
Os concretos do Observer não levam sufixo: o papel está na classe base, e
`ConsoleLoggerObserver` só repetiria a informação.

**Consequência visível.** A seção `method:` do YAML passou a ser `strategy:`, e
`src/training/callbacks.py` virou `src/training/observers.py`.

**O que não foi renomeado.** A pasta `methods/`, porque `CLAUDE.md` a prescreve na
estrutura de referência — e a divisão fica coerente: a pasta é o vocabulário do
domínio ("o método da sua trilha", no enunciado), a classe é o do padrão. Objetos
de valor (`StepOutput`, `EpochRecord`, `EvaluationResult`, `CheckpointState`,
`RunContext`) também ficaram: não são participantes de padrão e um sufixo neles
seria ruído.

---

## D6 — Pacote direto em `src/`, sem camada `pa3/` (08/11)

**Decisão.** Os módulos ficam em `src/` e os imports são `from src.data import ...`,
como no projeto de referência. `CLAUDE.md` foi alinhado ao repositório.

**Consequência.** O repositório deixa de ser um pacote instalável: não há
`[build-system]` no `pyproject.toml`, `uv sync` instala apenas as dependências e
tudo roda da raiz (`python scripts/train.py`, `python predict.py`). A raiz entra no
`sys.path` por `scripts/_bootstrap.py` e pelo cabeçalho do `predict.py`, para
funcionar no Colab e numa máquina limpa sem instalação.

**Alternativa descartada.** Manter `src/pa3/` para que o projeto fosse instalável
com `pip install -e .`. Nada no fluxo de entrega depende disso: o enunciado pede
um comando que treina, um que avalia e o `predict.py`, todos executados da raiz.

---

## Em aberto

- **Trilha (A, B ou C).** Bloqueia Parte 1, Parte 2 e Parte 3. A estrutura é
  neutra entre as três: a trilha entra como um arquivo em `src/methods/`.
- **Backbone da Parte 0 (b).** Nada registrado em `src/models/registry.py` ainda.
  O teto de 12M já é verificado na construção.
- **Política de aumentação.** Nada registrado em `src/data/augment.py` ainda;
  `scripts/` já pede a política `eval` para validação e teste.
