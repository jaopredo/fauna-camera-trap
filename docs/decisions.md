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


## D8 — ResNet-18 como backbone do PA (12/11)

**Decisão.** `resnet18` com pesos de ImageNet é a versão (b) da Parte 0 e o backbone
do resto do trabalho. A CNN do zero da versão (a) é a `small_cnn`, de 394.293
parâmetros.

**Fontes consultadas.**
- Enunciado (`docs/PA3.pdf`, seção 3 e Parte 0): o teto de 12 M "comporta ResNet-18,
  MobileNetV3 e EfficientNet-B0"; o teto de referência de **0,686 de F1 macro** foi
  medido pelo monitor com uma **ResNet-18**.
- Aula 13 (`DL_20242_13_Unsupervised.pdf`, **p. 60**): o experimento do professor usa
  "ResNet18 as backbone architecture".
- Aula 15 (`DL_20242_15_VisualFM.pdf`, **p. 15**): receita de fine-tuning — remover a
  cabeça pré-treinada, acoplar uma camada nova com o número de classes da tarefa,
  treinar nos dados disponíveis.

**Contagens medidas** (não estimadas — `src/models/budget.py` conta na construção):

| Modelo | Parâmetros com 21 classes | % do teto |
|---|---:|---:|
| `small_cnn` | 394.293 | 3,3% |
| `resnet18` | **11.187.285** | **93,2%** |
| EfficientNet-B0 (descartado) | ~4,03 M | ~34% |
| MobileNetV3-Large (descartado) | ~4,23 M | ~35% |

**Por que ResNet-18 e não as outras duas.** O enunciado manda usar 0,686 como a linha
do teto na Parte 2. Com outro backbone, essa linha deixaria de ser comparação direta —
a diferença entre a nossa curva e o teto passaria a misturar o efeito dos rótulos com
o efeito da troca de arquitetura. As alternativas são ~3× menores e mais rápidas, e
ficam registradas caso o custo em CPU se torne o gargalo.

**Risco aceito.** 93,2% do teto deixa pouca folga: qualquer acréscimo ao modelo
entregue precisa ser medido. `tests/test_models.py` trava o número exato, de modo que
uma mudança futura falha antes do treino, e não na conferência da entrega.

**Divergência consciente da aula.** A aula 15 (p. 15) manda inicializar a camada nova
com zeros. Isso está prescrito para ViT (Dosovitskiy 2021); mantivemos a
inicialização padrão do PyTorch para a `fc` da ResNet, por não haver na aula
evidência para CNN. Se a curva de validação mostrar instabilidade nas primeiras
épocas, o zero-init é a primeira coisa a testar — no `val`.

**Fora do material da disciplina.** Taxa de aprendizado (1e-4 no fine-tuning, 1e-3 no
treino do zero), otimizador (Adam), tamanho de batch (16) e número de épocas (60) não
têm respaldo nas aulas disponíveis — ver `docs/aulas_index.md`, seção "o que não está
nas aulas". São valores padrão, a serem ajustados **apenas no `val`** se necessário.

---

## D9 — Redimensionar em vez de recortar (12/11)

**Decisão.** As duas estratégias de aumentação (`eval` e `basic`, em
`src/data/transforms.py`) fazem `Resize((224, 224))`, distorcendo a proporção. Nada de
`CenterCrop` ou `RandomResizedCrop` na Parte 0.

**Por quê.** O enunciado diz que em um quarto das imagens o animal ocupa menos de 2%
do quadro, e que as proporções variam de câmera para câmera porque a faixa de texto
foi cortada do original. Um recorte central descartaria as bordas — e com elas o
animal, em boa parte dos casos. Não há proporção comum a preservar, então distorcer é
o preço mais barato. `tests/test_transforms.py` verifica que os quatro cantos da
imagem sobrevivem à transformação.

**Aumentação de treino: só espelhamento horizontal.** O item 3 da Parte 0 pergunta em
que época a rede começa a decorar as 105 imagens; aumentação forte adia a memorização
e embaralha exatamente essa leitura. Nenhuma das 21 espécies depende de lateralidade.
Aumentação agressiva é assunto do experimento obrigatório da Trilha A — e é lá que o
recorte que corta o animal de 2% do quadro deve ser medido, não assumido.

---

## D10 — As seeds são configuração, não shell (13/11)

**Decisão.** `ExperimentConfig.seed: int` virou `seeds: tuple[int, ...]`, declarado no
YAML (`seeds: [0, 42, 123]`). Um **experimento** declara o conjunto de seeds; uma
**execução** é uma delas. `scripts/train.py` varre as declaradas, cada uma no seu
diretório.

**Por quê.** As seeds estavam escritas à mão em dois lugares (`Makefile` e `README`) e
em nenhum deles no código — número mágico, que o `CLAUDE.md` proíbe, e duplicado, livre
para divergir. Pior: a regra do enunciado ("3 seeds no orçamento de 5, uma nos de 1 e
20") não estava registrada em lugar nenhum, então a Parte 2 poderia rodar outro
conjunto sem que nada acusasse — e a comparação com o ponto de 5 que já está no
`curva_orcamento.csv` quebraria em silêncio.

**O que mudou junto, para a fronteira ficar coerente:**

| Antes | Depois |
|---|---|
| `for s in 0 42 123` no Makefile e no README | os YAMLs declaram; nenhum laço de shell prescreve seed |
| f-string `f"{name}_b{budget}_s{seed}"` no `train.py` | `ExperimentConfig.run_name(seed)`, fonte única da convenção |
| `report_p0` lia `config.seed` | lê `manifest["run"]["seed"]` — o config declara o conjunto, o diretório é uma seed |
| `--figuras-seed` com padrão `0` fixo | padrão = primeira seed declarada (com `seeds: [7, 11, 13]`, o `0` não casaria e as figuras sumiriam em silêncio) |

**Validações acrescentadas:** lista vazia, seeds repetidas (sobrescreveriam o mesmo
diretório e falseariam a média ± desvio), valores não numéricos, e `seed:` no singular,
que agora falha dizendo o que escrever no lugar.

**Compatibilidade.** Os 6 manifestos das execuções da Parte 0 guardavam `seed`
singular. `src/tracking.py:read_manifest` traduz na leitura, sem reescrever o arquivo:
o registro do experimento fica intacto e as 38 min de treino continuam válidas. A seed
efetiva sempre esteve em `run.seed`, então a tradução não muda número nenhum.

---

## D11 — `torch` fora do `pyproject.toml` (13/11)

**Decisão.** `torch`, `torchvision` e `timm` saíram das dependências declaradas.
`make setup` roda `uv sync` e, em seguida,
`uv pip install torch torchvision --torch-backend=auto`.

**Por quê.** A roda correta do PyTorch depende da placa da máquina, e o índice de
onde ela vem faz parte da instalação, não da especificação do projeto. Com `torch`
declarado no `pyproject.toml`, o `uv` resolvia para `+cu132` em qualquer máquina — o
que nesta, com uma **Radeon RX 7600 (gfx1102) e ROCm 7.14**, instalava uma roda CUDA
inútil e o treino caía para CPU em silêncio.

`--torch-backend=auto` detecta o hardware. Verificado nesta máquina:

```
+ torch==2.14.1+rocm7.2
+ torchvision==0.29.1+rocm7.2
+ triton-rocm==3.8.0
```

e depois `make ambiente` confirma `GPU disponivel: True | AMD Radeon RX 7600`.

**Efeitos colaterais, todos desejáveis.**

- `timm` saiu junto: não é usado em lugar nenhum do código e era o único pacote que
  puxava `torch` por transitividade — mantê-lo anularia o efeito de tirar o torch.
  Se um backbone do `timm` for preciso depois, volta como uma linha.
- O `requirements.txt` versionado caiu de 198 para 77 linhas, sem `torch`, `triton`
  nem as ~20 bibliotecas `nvidia-*`. Os alvos `setup-pip` instalam o torch em
  seguida, como os com `uv`.
- `make setup-cpu` força a roda de CPU, que é o que o Colab sem GPU e a medição de
  tempo de inferência do modelo entregue precisam.
- `make ambiente` mostra qual roda ficou instalada e se a GPU foi encontrada —
  barato, e evita descobrir no meio de um treino de horas que ele está em CPU.

**Origem.** Mesmo esquema do projeto `object-detection-in-videos`, já usado por nós.

---

## Em aberto

- **Trilha (A, B ou C).** Bloqueia Parte 1, Parte 2 e Parte 3. A estrutura é
  neutra entre as três: a trilha entra como um arquivo em `src/methods/`.
- **Hiperparâmetros da Parte 0.** lr, batch e épocas são valores padrão sem
  respaldo nas aulas disponíveis; se forem ajustados, é no `val`.
