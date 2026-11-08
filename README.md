# PA3 — Poucos rótulos: fauna em câmeras trap

Classificação de 21 espécies do ENA24 com 5 rótulos por classe e 6.505 imagens de
treino sem rótulo. O produto final é um modelo de no máximo 12 milhões de
parâmetros que roda em CPU, sem internet.

## Ambiente

Python 3.12 (a versão do Colab). Com [uv](https://docs.astral.sh/uv/):

```bash
make setup          # uv sync
```

Sem uv:

```bash
make setup-pip      # venv + pip install -r requirements.txt
```

## Dados

O pacote `pa3_dados.zip` não está no repositório (415 MB). Descompacte-o em
`data/`, de modo a ficar:

```
data/classes.csv
data/labels/{val,test_public,rotulos_1,rotulos_5,rotulos_20}.csv
data/images/{train,val,test_public}/<image_id>.jpg
```

Os testes que dependem do pacote são pulados se ele não estiver presente.

## Um comando que treina

```bash
python scripts/train.py --config configs/exemplo.yaml
```

Variações sem criar outro YAML (é assim que a Parte 2 varre seeds e orçamentos):

```bash
python scripts/train.py --config configs/exemplo.yaml --set seed=1 data.budget=20
python scripts/train.py --config configs/exemplo.yaml --resume
```

Cada execução grava em `outputs/<nome>_b<orçamento>_s<seed>/`: `manifest.json`
(configuração, seed e commit), `history.json`, `best.pt` e `last.pt`.

## Um comando que avalia

```bash
python scripts/evaluate.py --config configs/exemplo.yaml \
    --checkpoint outputs/<run>/best.pt --split val
```

`--split val` é o conjunto de seleção de modelo; `--split test_public` só reporta
números. A regra está no código, não só na documentação: as funções de seleção em
`src/evaluation/selection.py` recusam qualquer resultado que não venha do
`val`.

## Predição (entrega)

```bash
python predict.py --imagens PASTA --saida preds.csv
```

Escreve `image_id,label` com os nomes de `classes.csv`, roda em CPU, carrega
somente pesos locais e imprime a contagem de parâmetros do modelo.

## Testes

```bash
make test
```

## Estrutura

```
configs/              experimentos em YAML
src/
  config/             schema congelado e carregamento
  data/               catálogo, splits, datasets, Strategy de aumentação, loaders
  models/             Factory com Registry de modelos, teto de 12M, checkpoints
  methods/            Strategy de treino (uma trilha = uma implementação)
  training/           laço de treino e observadores (Observer)
  evaluation/         avaliação e seleção de modelo
  metrics.py          F1 macro, acurácia, matriz de confusão
  analysis/           figuras e diagnósticos das Partes 1 a 3
  tracking.py         seed, commit e manifesto de cada execução
scripts/              pontos de entrada; é onde os componentes são compostos
tests/
docs/                 decisions.md, STATUS.md e o enunciado
predict.py
```

## Padrões de projeto

Cada padrão aqui existe porque o enunciado exige **mais de uma implementação** do
mesmo papel. Onde não há variação prevista, não há padrão: `metrics.py` são funções
puras, `config/` e `tracking.py` são dataclasses congeladas.

| Padrão | Onde | Papel que varia | O que o enunciado exige |
|---|---|---|---|
| Strategy | `src/methods/` | método de treino | Parte 0 supervisionado + o método da trilha (Parte 1) |
| Strategy | `src/data/augment.py` | aumentação | experimento obrigatório da Trilha A; par fraco/forte do FixMatch |
| Factory com Registry | `src/models/registry.py` | arquitetura | 2 modelos na Parte 0; 3 backbones lado a lado na Trilha A |
| Observer | `src/training/observers.py` | log e checkpoint | registrar cada execução sem acoplar ao laço de treino |

### Strategy — método de treino

**Onde.** Contrato em `src/methods/base.py`; seleção em `src/methods/registry.py`;
consumido por `Trainer` em `src/training/loop.py`.

**Como.** `TrainingStrategy` é uma ABC com um método abstrato e três ganchos
opcionais:

```python
class TrainingStrategy(ABC):
    required_sources: tuple[str, ...] = ()   # "labeled", "unlabeled"
    primary_source: str = ""                 # define o tamanho da época

    @abstractmethod
    def training_step(self, model, batch, *, device) -> StepOutput: ...

    def on_epoch_start(self, epoch: int) -> None: ...      # rampa de peso, τ por classe
    def state_dict(self) -> dict[str, Any]: ...            # vai para o checkpoint
    def load_state_dict(self, state) -> None: ...          # volta no --resume
```

O laço **não sabe qual estratégia está rodando**. Ele pergunta de quais fontes de
dados ela precisa, monta o batch como `{"labeled": ..., "unlabeled": ...}` e pede a
perda. Quem chama `backward()` e o otimizador é o laço, nunca a estratégia.

**Por que importa aqui.** É o que permite FixMatch (105 rótulos contra 6.505 imagens
sem rótulo) coexistir com o supervisionado sem nenhum `if` no laço: a fonte primária
define o tamanho da época e as demais são cicladas, senão uma época terminaria em 4
batches.

**Como adicionar a trilha escolhida** — um arquivo novo, nenhuma edição no laço:

```python
# src/methods/fixmatch.py
@register_strategy("fixmatch")
def build_fixmatch(*, num_classes: int, tau: float = 0.95) -> TrainingStrategy:
    return FixMatch(num_classes=num_classes, tau=tau)
```

Depois importe o módulo em `src/methods/__init__.py` (para o decorador rodar) e
troque `strategy.name` no YAML. `tests/test_strategy_contract.py` passa a rodar
sobre ela **automaticamente**, por `available_strategies()`.

### Strategy — aumentação

**Onde.** `src/data/augment.py`. Selecionada por `data.augment` no YAML.

**Como.** Uma estratégia é um chamável `imagem PIL -> entrada do modelo`; para duas
vistas (contrastivo, fraco/forte) devolve uma tupla. O registro é por decorador, e o
construtor recebe o que o YAML declara mais o que só o chamador sabe:

```python
build_augmentation(config.data.augment, image_size=config.data.image_size)
```

**Por que não um `if` no dataset.** A Trilha A tem de medir duas aumentações lado a
lado (o *random resized crop* que, num animal ocupando 2% do quadro, gera um par
positivo sem animal nenhum). Com o `if` no dataset, esse experimento viraria uma
edição no caminho de leitura de dados; aqui é uma linha no YAML.

### Factory com Registry — arquitetura

**Onde.** `src/models/registry.py`. Selecionada por `model.name` no YAML.

**Como.** `register_model` é um decorador; `build_model` resolve o nome, instancia e
**verifica o teto de 12 milhões de parâmetros na construção**:

```python
model = build_model(config.model, num_classes=num_classes)
```

A verificação é o ponto principal de passar pela factory: um modelo acima do teto
treinado por horas não pode ser entregue, e o erro só apareceria na conferência
final. Há uma saída explícita, `enforce_budget=False`, para o professor da Trilha C,
que não faz parte do modelo entregue.

**Nome desconhecido falha listando o que existe**, em vez de cair num padrão
silencioso:

```
erro: model.name desconhecido: 'TBD'. Disponiveis: []
```

### Observer — log e checkpoint

**Onde.** `src/training/observers.py`. O `Trainer` recebe os observadores por
injeção, em `observers=`.

**Como.** O laço não imprime nem grava nada: emite três eventos.

```python
class TrainingObserver:
    def on_train_start(self, context: TrainingContext) -> None: ...
    def on_epoch_end(self, record: EpochRecord, context: TrainingContext) -> None: ...
    def on_train_end(self, history, context: TrainingContext) -> None: ...
```

Os ganchos são vazios na base, então cada observador sobrescreve só o que usa (o
**I** do SOLID). Três concretos, compostos em `scripts/train.py`:

| Observador | O que faz |
|---|---|
| `ConsoleLogger` | uma linha por época no terminal |
| `HistoryWriter` | `history.json`, reescrito a cada época — se o treino cair, as épocas anteriores continuam em disco |
| `CheckpointWriter` | `best.pt` (o escolhido no `val`) e `last.pt` (retomável) |

`best.pt` e `last.pt` são arquivos diferentes de propósito: o melhor modelo pode ser
de uma época bem anterior à última. `CheckpointWriter` decide o "melhor" chamando
`is_improvement`, que **recusa resultado que não venha do `val`** — a barreira do
`test_public` alcança até o checkpoint.

**Testável sem disco:** o laço é verificado em `tests/test_training_loop.py` com um
modelo e uma estratégia falsos, e sem observador nenhum quando o teste só olha o
histórico.

### Padrões considerados e recusados

| Padrão | Por que não |
|---|---|
| **Singleton** para configuração | É o que o projeto de referência faz, e a segunda chamada com outro YAML devolve silenciosamente a primeira configuração. A Parte 2 roda 3 seeds × 3 orçamentos: as três seeds rodariam iguais, sem nada no log. `load_config` devolve um objeto novo a cada chamada |
| **Adapter** para o professor externo | Faria sentido na Trilha C (interface comum para CLIP/BioCLIP), mas a trilha não foi escolhida. Criar agora seria abstração para zero implementações |
| **Template Method** no laço de treino | Exigiria herdar do `Trainer` para variar o método. Strategy mantém o laço fechado e a variação por composição |
| Hierarquia para métricas | Só há uma forma de calcular F1 macro. São funções puras em `src/metrics.py` |

## Documentação

- Seção **Padrões de projeto** acima — onde cada padrão está e como estender.
- `docs/STATUS.md` — feito, próximo passo, riscos e marcos.
- `docs/decisions.md` — decisões de arquitetura e modelagem, com as alternativas
  descartadas.
- `AI_LOG.md` — como a IA foi usada no trabalho.
