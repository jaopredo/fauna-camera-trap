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

## D12 — FixMatch: τ não é o do paper (13/11)

**Decisão.** O padrão do projeto é **τ = 0,80**, não o 0,95 do FixMatch original, e a
escolha final sai de uma varredura no `val`.

**Fonte.** Aula 13 (`DL_20242_13_Unsupervised.pdf`), p. 85–91: regularização de
consistência mais pseudo-rotulagem (p. 85); razão μ e peso λ como entradas (p. 87);
CE supervisionada na vista fraca (p. 88); pseudo-rótulo calculado na vista fraca e
versão forte guardada (p. 89); perda não supervisionada com **pseudo-rótulo duro** e
**limiar de ativação** (p. 90); soma ponderada (p. 91).

**Por que divergir do paper.** Medimos o pseudo-rotulador inicial — o modelo da
Parte 0 — no `val`, média sobre as 3 seeds:

| τ | cobertura | precisão | classes vivas (de 21) |
|---|---|---|---|
| 0,70 | 20,1% | 78,0% | 12,0 |
| 0,80 | 17,3% | 77,8% | 10,3 |
| 0,95 | 8,4% | 89,0% | **6,0** |

O τ = 0,95 foi calibrado em **CIFAR-10**: 10 classes balanceadas e um modelo
confiante. Aqui, com 21 classes desbalanceadas e 105 rótulos, ele deixa **6 das 21
classes sem nenhum pseudo-rótulo** — o colapso de classe que o enunciado manda
investigar, já na largada. É exatamente o alerta do `CLAUDE.md`: resultado em CIFAR-10
não se transfere para câmeras trap.

**Outras escolhas, com o porquê:**

- **`primary_source = "unlabeled"`.** Uma época varre as 6.505 imagens e cicla os 105
  rótulos. Ao contrário, a época acabaria em 7 lotes e o pool quase não seria visto.
- **Um forward só** para rotulados e vista forte concatenados. Além de mais rápido,
  mantém as estatísticas de BatchNorm coerentes: dois forwards separados fariam a BN
  ver dois lotes de distribuições diferentes e deslocariam as médias acumuladas.
- **Denominador fixo em μ·B** na perda não supervisionada, como no paper. Dividir
  pelo número que passou faria o peso do termo crescer sozinho conforme o modelo
  ganha confiança, e λ deixaria de significar o mesmo ao longo do treino.
- **`mask_rate` nos componentes**, gravado por época no `history.json`. É o sinal que
  diz se o FixMatch tem matéria-prima; perto de zero significa treino sem efeito, e
  isso não apareceria em nenhum outro número.

---

## D13 — Alinhamento de distribuição com alvo uniforme (13/11)

**Decisão.** A correção de viés exigida pela Trilha B é o **alinhamento de
distribuição**: mantém média móvel $\tilde p$ das predições na vista fraca e realinha
$q' = \mathrm{normalize}(q \cdot p_{\text{alvo}} / \tilde p)$ antes do limiar. O alvo
é **uniforme** sobre as 21 classes.

**Por que uniforme.** Três razões, nesta ordem:

1. O orçamento de rótulos é uniforme **por construção** — `rotulos_5.csv` tem
   exatamente 5 imagens por classe. É a única distribuição que conhecemos de fato.
2. Não dá para conhecer o prior do pool sem rótulo sem rotulá-lo, que é precisamente
   o que não podemos fazer.
3. A métrica principal é o **F1 macro**, que pesa cavalo igual a urso. Empurrar os
   pseudo-rótulos para o balanceado é alinhado à métrica pela qual somos avaliados,
   mesmo que desalinhado do prior verdadeiro (690 ursos contra 45 cavalos no train).

**Ressalva honesta para a apresentação:** o alvo uniforme está errado em relação ao
mundo. Ele é uma escolha deliberada a favor das classes raras, e é assim que deve ser
apresentado — não como estimativa do prior.

**Fora do material da disciplina.** O enunciado oferece a técnica pelo nome, mas ela
não aparece nas aulas 11–15 (ver `docs/aulas_index.md`). A formulação vem da
literatura de ReMixMatch/FixMatch.

**Estado.** A média móvel vive em `state_dict()`/`load_state_dict()` da estratégia, que
o checkpoint já grava em `strategy_state`. Sem isso, `--resume` reiniciaria o
realinhamento em uniforme e o modelo levaria centenas de passos para reconstruir a
estatística.

---

## D14 — Controle do confundidor de aumentação (13/11)

**Decisão.** `configs/p0_finetune_weak.yaml` roda o piso com a mesma aumentação fraca
do FixMatch (`weak`: espelhamento + translação de 12,5%), uma seed.

**Por quê.** A Parte 0 usou `basic` (só espelhamento) e a Trilha B usa `weak`. Sem
esse controle, parte de qualquer ganho do FixMatch poderia vir da aumentação, e não
dos dados sem rótulo — e não haveria como separar os dois na apresentação. Custa ~5
min de GPU e transforma uma dúvida numa medida.

---

## D15 — Um forward só sobre os três ramos (13/11)

**Decisão.** `FixMatchStrategy.training_step` faz **um** forward sobre
`cat([rotulado, fraca, forte])`, em vez de dois forwards separados.

**Por que era um defeito, e não uma otimização.** Com forwards separados:

1. a BatchNorm **atualiza as estatísticas acumuladas duas vezes por passo**, com
   duas distribuições diferentes (o lote misto e o lote só de vistas fracas);
2. o pseudo-rótulo sai calculado sob as estatísticas de lote da vista fraca, que
   não são as mesmas usadas na vista forte contra a qual ele é aplicado — alvo e
   predição passam a viver em normalizações distintas.

**Efeito medido** nas três primeiras épocas, mesma seed e mesmo τ:

| | época 2 | época 3 |
|---|---|---|
| dois forwards | `mask_rate` 0,097 · val F1 0,257 | 0,233 · 0,311 |
| forward único | `mask_rate` 0,155 · **val F1 0,292** | 0,303 · **0,315** |

Ajuda, mas **não resolve** o problema central: o `ce` continua caindo para 0,03 na
época 3, e a espiral de superconfiança segue.

**Custo.** 240 imagens no grafo em vez de 128 → **6,2 GB de VRAM** (de 8,6
disponíveis) e 39 s por época em vez de 28 s. Cabe, com folga de 2,4 GB.

---

## D16 — lr 3e-5 no FixMatch: a mesma taxa não serve com 8× mais passos (14/11)

**Decisão.** `configs/p1_fixmatch*.yaml` usam `lr: 0.00003`, contra `0.0001` da
Parte 0. Escolhido no `val`.

**Por quê — a conta que explica o fracasso inicial.** Com `primary_source =
unlabeled`, uma época tem ~58 passos (6.505 ÷ 112) em vez dos 7 da Parte 0
(105 ÷ 16). Cada época cicla os 105 rótulos **~9 vezes**, e em 30 épocas são ~265
passagens pelo conjunto rotulado, contra 60 na Parte 0. A mesma taxa de aprendizado
passa a memorizar 4× mais.

**O que isso provocou,** medido, com lr = 1e-4:

| sintoma | valor |
|---|---|
| `ce` ao fim do treino | 0,0006 — memorização completa dos 105 rótulos |
| confiança média no `val` | 0,941 (o piso: 0,418) |
| `mask_rate` ao fim | 0,995 — **o limiar deixou de filtrar** |
| precisão do pseudo-rótulo (τ=0,80) | **36,5%** (o piso: 74,2%) |

O modelo ficou *confiantemente errado*: a confiança dobrou enquanto a precisão caía
pela metade. Treinando nos próprios pseudo-rótulos, ele se convence dos próprios
erros — viés de confirmação. E como a confiança deixou de correlacionar com acerto,
τ virou decoração: subir para 0,95 ainda deixava passar 83% das imagens.

**Com lr = 3e-5:** `mask_rate` final 0,625 (o limiar volta a filtrar) e melhor F1 de
validação 0,3856 contra 0,3548. Ainda há memorização (`ce` ≈ 0,002), mas lenta o
bastante para o limiar continuar significando alguma coisa.

**Ressalva honesta.** A grade de τ ({0,70; 0,80}) foi comparada com lr = 1e-4, e só
depois mudamos a lr. O τ escolhido pode não ser o ótimo para a lr nova. Refazer a
varredura custaria mais 40 min de GPU, que não cabem no prazo — fica registrado como
limitação, não como detalhe omitido.

**Para a apresentação.** Este é o achado mais transferível do trabalho: ao trocar o
`primary_source` para o conjunto não rotulado, o número de passos por época muda de
ordem de grandeza, e **todo hiperparâmetro calibrado por época deixa de valer**. Não
é um detalhe do FixMatch; vale para qualquer método semi-supervisionado montado sobre
um baseline supervisionado.

---

## D17 — O resultado da Parte 1 é negativo em F1 e positivo no diagnóstico (14/11)

**Decisão.** Reportar o FixMatch como **resultado negativo** no F1 macro e manter o
piso supervisionado como modelo entregue. Nenhum número é maquiado, nenhuma execução
é descartada.

**Os números** (`test_public`, 3 seeds, orçamento de 5; época escolhida no `val`):

| Método | F1 macro | Acurácia | vs. piso |
|---|---|---|---|
| `piso_finetune` | 0,465 ± 0,038 | 0,489 ± 0,028 | — |
| `piso_aug_weak` (controle) | 0,482 (1 seed) | 0,502 | **+0,017** |
| `fixmatch` | 0,438 ± 0,005 | 0,490 ± 0,025 | **−0,027** |
| `fixmatch_da` | 0,438 ± 0,033 | 0,481 ± 0,043 | **−0,027** |
| teto de referência | 0,686 | — | — |

O FixMatch fica **abaixo do piso** e abaixo do controle de aumentação. O DA não
muda a média e triplica o desvio entre seeds (0,005 → 0,033).

**Por que isso não é um bug.** Quatro causas foram testadas e descartadas por
medição antes de concluir:

1. **BatchNorm** — dois forwards separados atualizavam as estatísticas duas vezes por
   passo e calculavam o pseudo-rótulo sob normalização diferente da vista forte
   (D15). Corrigido: um forward sobre os três ramos.
2. **Limiar** — τ do paper (0,95) deixa 6 das 21 classes sem pseudo-rótulo; a grade
   {0,70; 0,80} foi varrida e escolhida no `val` (D12).
3. **Taxa de aprendizado** — a mesma lr da Parte 0 memoriza 4× mais com
   `primary_source = unlabeled`; corrigida para 3e-5 (D16).
4. **Confundidor de aumentação** — o controle `piso_aug_weak` isola o efeito da
   aumentação fraca (D14), e é ele que explica o único ganho observado.

**O diagnóstico obrigatório 2 contraria a previsão do enunciado.** O enunciado diz
que "os pseudo-rótulos tendem a ir para as classes grandes". No nosso treino, **não
vão**. A correlação de Pearson entre o suporte real da classe e a razão de viés
(fatia dos pseudo-rótulos ÷ fatia real) é de **+0,125** no piso — praticamente nula:

| classe | imagens no `test_public` | razão de viés (piso) |
|---|---:|---:|
| `american_black_bear` (a maior) | 82 | **0,33** — sub-representada |
| `eastern_chipmunk` | 31 | **4,21** — super-representada |
| `virginia_opossum` | 55 | 1,97 |
| `bird` | 22 | 0,00 — nenhum pseudo-rótulo |

O viés é **por confiança**, não por frequência: o pseudo-rotulador favorece as
classes visualmente distintivas (o esquilo listrado é inconfundível) e abandona as
que se confundem entre si, independentemente do tamanho. A classe mais frequente do
conjunto recebe três vezes menos pseudo-rótulos do que mereceria.

**O FixMatch melhorou o pseudo-rotulador mesmo perdendo em F1.** Em τ=0,80, no
`test_public`:

| | piso | `fixmatch` | `fixmatch_da` |
|---|---:|---:|---:|
| cobertura | 22,2% | **41,7%** | 34,9% |
| precisão agregada | 95,6% | 86,1% | 84,5% |
| classes vivas (de 21) | 14 | **19** | 19 |
| classes com zero pseudo-rótulos | **7** | 2 | 2 |
| desvio da razão de viés | 1,12 | **0,88** | 0,91 |

O colapso de classe **foi corrigido** — de 7 classes mudas para 2 —, a cobertura
quase dobrou e a dispersão do viés caiu. O preço foi precisão: 95,6% → 86,1%. Em F1
macro o saldo é negativo, mas a afirmação "o FixMatch não fez nada" seria falsa.

**O DA fez o que promete e não bastou.** Ele inverteu o sinal da correlação entre
suporte e viés (+0,111 → **−0,117**): com alvo uniforme, passou a empurrar
pseudo-rótulos para as classes pequenas, e ressuscitou `eastern_cottontail` e
`horse`. Só que a causa do problema não era a frequência — era a confiança. Corrigir
a distribuição marginal não corrige um pseudo-rotulador que erra com confiança alta.
**É a conclusão mais honesta do trabalho: a correção estava alinhada com o
diagnóstico do enunciado, não com o diagnóstico que medimos.**

**Alternativa descartada por prazo.** Limiar por classe atacaria a causa certa
(exigir mais confiança de `eastern_chipmunk` e menos de `american_black_bear`), mas
precisaria de nova varredura no `val` e de 6 execuções novas — ~2 h de GPU que não
cabem antes de 16/11. Fica registrado como o próximo passo, não como algo esquecido.

**Fontes.** Aula 13 (`DL_20242_13_Unsupervised.pdf`, p. 85–91) para o FixMatch; DA
está **fora do material da disciplina** (o enunciado o oferece pelo nome, mas ele não
aparece nas aulas 11–15).

**Para a apresentação.** Um resultado negativo bem medido vale mais do que um ganho
que ninguém sabe explicar — e o controle de aumentação prova que sabemos de onde veio
cada décimo. A pergunta provável da banca é "e se o FixMatch tivesse funcionado?", e
a resposta é a tabela do pseudo-rotulador: ele funcionou como pseudo-rotulador e
falhou como treino.

---

## D18 — Uma pasta por trilha, com a independência verificada por teste (14/11)

**Decisão.** Cada trilha vive em `src/tracks/<trilha>/` e é removível apagando a
pasta, os configs e os testes dela — sem editar nenhum arquivo compartilhado.
`src/tracks/__init__.py` descobre as subpastas com `pkgutil` e as importa.

**O problema concreto.** Antes disso a Trilha B não era isolada: ela vazava para
`src/analysis/figures.py` (duas funções), `src/data/transforms.py` (`weak_strong`) e
`src/methods/__init__.py` (a linha de import que faz o decorador rodar). Apagar a
Trilha B exigiria editar três arquivos compartilhados, e implementar a Trilha C em
cima disso dobraria o acoplamento.

**Por que import implícito, e só aqui.** Em todo o resto do projeto os imports são
explícitos. Com `from src.tracks import b, c`, apagar a Trilha C quebraria o import
da B — exatamente o que se queria evitar. A descoberta fica confinada à única pasta
cujo propósito é ser ponto de extensão.

**O que ficou de fora da pasta, de propósito:**

- a aumentação `weak` — o modelo entregue a usa, então é infraestrutura. Apagar a
  Trilha B não pode levar embora a aumentação do modelo que entregamos. O par
  `weak_strong`, esse sim, é da trilha;
- `src/analysis/budget.py` — a curva da Parte 2 é comum às duas trilhas;
- `src/evaluation/runs.py` — **extraído nesta decisão**. "Reabrir uma execução salva
  e pegar as probabilidades dela" já estava duplicado entre `report_p0.py` e
  `report_p1.py`; sem um lugar comum, o relatório da Trilha C teria de importar do
  relatório da Trilha B. A extração removeu **10 imports órfãos** do `report_p1.py`,
  medida de quanta duplicação havia.

**A independência é asserção, não promessa.** `tests/test_tracks_independence.py`
percorre a árvore sintática (`ast`, não `import`: um import dentro de função passaria
batido) e falha se uma trilha importar de outra, se a infraestrutura importar
qualquer trilha, ou se um script neutro depender de uma. Verificado de verdade:
removendo `src/tracks/b/`, 189 testes continuam passando.

**Consequência no teste de contrato.** A destilação da Trilha C precisa do `.npz` do
professor para nascer, e o teste comum não sabe montá-lo. Em vez de uma lista de
exceções no teste compartilhado — que ficaria obsoleta ao apagar uma trilha —, o
contrato aceita duas formas: a estratégia nasce e cumpre o contrato, **ou** recusa
com uma mensagem que diz o que falta e como obter. O contrato em si é verificado pelo
teste da própria trilha. Isso expôs um defeito real: sem `cache`, a fábrica dava
`TypeError` cru para quem estava escrevendo um YAML.

---

## D19 — Trilha C: o professor é pior que o piso, e o especializado é pior que o genérico (14/11)

**Decisão.** Implementar a Trilha C ao lado da B, com dois professores zero-shot de
pesos abertos, e reportar que **os dois são piores que o piso supervisionado**.

**Os números** (`test_public`, zero-shot, prompt `"a photo of a {}."`):

| Professor | Pré-treino | F1 macro | Acurácia |
|---|---|---|---|
| `clip` | ViT-B/16, pares imagem-texto da web | **0,350** | 0,379 |
| `bioclip` | ViT-B/16, TreeOfLife-10M (biologia) | **0,313** | 0,363 |
| piso supervisionado (105 rótulos) | — | 0,465 | 0,489 |

Os dois são a mesma arquitetura, então a diferença mede o **pré-treino**, não o
tamanho. O especializado em biologia perde para o genérico — contra a expectativa.

**Calibração, medida antes de treinar qualquer aluno:**

| Professor | Confiança ao acertar | Ao errar | Separação |
|---|---|---|---|
| `clip` | 0,710 | 0,358 | **+0,352** |
| `bioclip` | 0,833 | 0,608 | +0,225 |

O `bioclip` é **confiantemente errado**: erra com confiança média de 0,61. O `clip` é
razoavelmente calibrado, e por isso é ele que rotula o `train` nos configs do aluno.

**O colapso de classe reaparece.** Em τ = 0,80 o `clip` mantém 20,5% das imagens com
92,2% de precisão, mas deixa só **13 das 21 classes** vivas. É a mesma patologia da
D17, agora num pseudo-rotulador de origem completamente diferente — o que reforça que
ela é do *limiar de confiança*, não do FixMatch.

**Previsão registrada antes de rodar os alunos.** Se o diagnóstico da D17 valer, o
tratamento `filter` perde de `soft` em F1 macro **apesar de treinar com rótulos mais
limpos**, porque filtra classe junto com incerteza. Os três configs existem para
falsificar isso. **Os alunos ainda não foram treinados.**

**Um bug que daria número plausível.** Os pesos da OpenAI foram treinados com
QuickGELU, e o `open_clip` tem arquiteturas separadas. Com `ViT-B-16` (sem sufixo) ele
carrega os pesos numa rede de GELU padrão, **avisa e roda**: medimos 0,384 antes de
notar o aviso, contra 0,350 com `ViT-B-16-quickgelu`, que é o correto. O número errado
era o mais bonito. Travado em `tests/test_track_c_teacher.py`.

**Fora do material da disciplina.** CLIP, SigLIP e BioCLIP não aparecem nas aulas
11-15, nem destilação professor-aluno. Busca por "clip", "siglip", "bioclip" e
"contrastive" nas aulas 12 e 15: a 15 cobre ViT, Swin, MAE, MSN, SAM e visual
prompting; a 12 cobre redes siamesas e perda contrastiva, não language-image. O
enunciado nomeia e autoriza estes modelos; as aulas não os ensinam.

**Conformidade com o enunciado,** item a item: o professor é de pesos abertos rodando
na máquina local (não é API nem modelo fechado — a proibição é sobre "API de LLM ou
de modelo fechado (GPT, Gemini, Claude e afins)", e o CLIP está na lista de
permitidos); vê só o `train` e o `test_public`, com o `val` recusado por exceção; e
não faz parte do modelo entregue, o que é **estrutural**: ele grava um `.npz` e o
aluno lê o arquivo, de modo que `predict.py` não tem caminho até ele.

**Sobre o prompt.** O template é fixo no módulo, não parâmetro de linha de comando.
Escolher entre templates seria seleção de modelo, e os dois conjuntos disponíveis
estão fechados: o `val` o professor não pode ver, o `test_public` é proibido para
escolha. Sem escolha, não há regra a interpretar. **Limitação registrada:** o BioCLIP
pode ser documentado para nomes taxonômicos, que não testamos — o 0,313 pode
subestimá-lo.

---

## D20 — A previsão da D19 foi falsificada: o filtro de confiança ganhou (14/11)

**Decisão.** O método entregue é o **aluno da Trilha C com filtro por confiança**.
A Trilha B fica no repositório como **registro de aprendizado**, e as Partes 2 e 3
são construídas sobre a Trilha C.

**O resultado** (`test_public`, orçamento de 5, 1 seed; época escolhida no `val`):

| Método | F1 macro `test_public` | F1 macro `val` |
|---|---:|---:|
| professor `bioclip` (zero-shot) | 0,313 | — |
| professor `clip` (zero-shot) | 0,350 | — |
| `aluno_sem_tratamento` | 0,312 | 0,3587 |
| `aluno_rotulos_suaves` | 0,351 | 0,4602 |
| **`aluno_filtro_confianca`** | **0,526** | **0,4675** |
| piso | 0,465 ± 0,038 | 0,3913 |
| controle `piso_aug_weak` | 0,482 | 0,4103 |
| teto de referência | 0,686 | — |

**0,526 é o melhor resultado do projeto**, e recupera 28% da distância entre o piso
e o teto. O `val` concorda com a ordenação, então a escolha é legítima e robusta.

**A previsão registrada na D19 estava errada.** Eu escrevi, antes de rodar: *"se o
diagnóstico da D17 valer, `filter` perde de `soft` apesar de treinar com rótulos mais
limpos"*. O `filter` ganhou por **+0,175**. Não foi erro de margem; foi erro de
mecanismo.

**Por que errei — a distinção que faltava.** O que importa não é o limiar, é **de
quem vem a confiança**:

| | Trilha B (FixMatch) | Trilha C (destilação) |
|---|---|---|
| fonte da confiança | o **próprio modelo**, em evolução | o **professor**, congelado |
| laço de realimentação | sim — ele se convence dos próprios erros | **não existe** |
| calibração | degrada durante o treino | fixa: separação +0,352 |

A lição correta, que substitui a da D17 como generalização:

> **Um filtro de confiança funciona quando a confiança vem de uma fonte externa,
> congelada e calibrada. Falha quando vem das predições em evolução do próprio
> modelo.**

Isso é mais afiado que a previsão original e **só ficou visível porque rodamos as
duas trilhas**. É o retorno concreto de ter feito B e C, e é o motivo de a B
permanecer no repositório.

**Soft preserva a incerteza; não remove o erro.** Quando o professor diz "raposa
cinza 0,31, raposa vermelha 0,28" e a imagem é um guaxinim, o alvo suave ensina uma
distribuição *errada com elegância*. O filtro faz outra coisa: **remove a imagem**.
Com um professor de 35% de acerto, remover o errado vale mais que preservar a dúvida
sobre o errado. Medido: `kept_rate = 0,225` — o aluno treinou em ~1.460 das 6.505
imagens, aquelas em que o professor acerta 92,2% em vez de 35%.

**Sem tratamento, o ruído passa direto.** `aluno_sem_tratamento` deu 0,312, contra
0,350 do professor. Era exatamente para isso que a execução sem tratamento existia.

**Obrigatório 3 — em quais classes o aluno supera o professor, e por quê.** Em **14
de 21**. E há um padrão, com correlação de Pearson **−0,608** entre a pureza do
pseudo-rótulo e o ganho do aluno:

| Pseudo-rótulo do professor | Classes | Ganho médio do aluno |
|---|---:|---:|
| sujo (pureza < 0,30) | 8 | **+0,296** |
| limpo (pureza > 0,70) | 6 | −0,038 |

**O aluno ganha mais onde o professor é pior.** Caso extremo: `eastern_chipmunk`,
professor 0,000 → aluno 0,967, com pureza 0,0 — o professor nunca acertou um esquilo
listrado, e o aluno aprendeu só dos 5 rótulos verdadeiros. Também `dog` (0,055 →
0,686) e `eastern_gray_squirrel` (0,338 → 0,877). A explicação: onde o rótulo do
professor é sujo, os 105 rótulos verdadeiros dominam e o ruído se dilui; onde ele já
era bom, não há o que acrescentar.

**O colapso de classe que eu previ aconteceu — em uma classe.** `bird` foi de 0,296
para **0,000**: o aluno não prevê mais essa classe. `domestic_cat` quase (0,286 →
0,080), apesar de pureza 1,0. O filtro guardou pouquíssimas imagens confiantes
dessas classes e o aluno ficou sem sinal. A patologia é real, e foi compensada pelos
ganhos nas outras 19.

**Correção de integridade feita junto.** O `report_p1c.py` escolhia qual aluno
desenhar na figura pelo F1 de **`test_public`**. Não mudou número nenhum (o `val`
concorda), mas a justificativa estava errada. Agora sai de
`src/evaluation/runs.py:best_validation_metric`, que lê `val_<monitor>` do
`history.json` — e que **não consegue** devolver métrica de teste, porque o laço de
treino nunca avalia no `test_public` e portanto não existe chave de teste no arquivo.
Garantia estrutural, com teste em `tests/test_runs.py`.

**Para a apresentação.** A sequência é a história: a D17 produziu um diagnóstico, a
D19 virou esse diagnóstico em previsão falsificável, e a D20 a refutou e corrigiu o
mecanismo. Previsão registrada antes do experimento, experimento rodado, previsão
refutada, mecanismo revisado — e nada disso foi reescrito depois para parecer certo.

---

## D21 — Parte 3: CAM em vez de Grad-CAM, e a oclusão como teste do mapa (14/11)

**Contexto.** O enunciado pede, para cada erro, "um mapa de saliência (Grad-CAM ou
oclusão) que mostre para onde o modelo olhou".

**Opções consideradas.**

| Opção | Fonte | Custo | Risco |
|---|---|---|---|
| **CAM** (escolhida) | `docs/DL_20242_13_Unsupervised.pdf`, **p. 80–82**, "Weakly-Supervised Learning" (Zhou et al., CVPR 2016); também no `A2.pdf` do aluno, §3.6.1 | nenhum: os pesos já existem | exige GAP seguido de linear |
| Grad-CAM | fora do material da disciplina | hook de gradiente, backward por imagem | mais peças para dar errado |
| Oclusão por varredura | citada no enunciado | ~196 forwards por imagem | caro e de baixa resolução |

**Decisão: CAM.** E a justificativa é mais forte que "está na aula": **nesta
arquitetura, CAM e Grad-CAM são o mesmo mapa.** Com $A_k$ o $k$-ésimo mapa de
ativação, $Z$ o número de posições e $w_k^c$ os pesos da `fc`:

$$y^c = \sum_k w_k^c \cdot \frac{1}{Z}\sum_{ij} A_k(i,j)
\;\Longrightarrow\;
\frac{\partial y^c}{\partial A_k(i,j)} = \frac{w_k^c}{Z}
\;\Longrightarrow\;
\alpha_k^c = \frac{1}{Z}\sum_{ij}\frac{\partial y^c}{\partial A_k(i,j)} = \frac{w_k^c}{Z}$$

Logo $\mathrm{ReLU}(\sum_k \alpha_k^c A_k) = \frac{1}{Z}\,\mathrm{ReLU}(\sum_k w_k^c A_k)$:
os dois mapas diferem por uma constante positiva, e depois da normalização pelo pico
são **idênticos**. Não é analogia — `tests/test_saliency.py` calcula o Grad-CAM pelo
autograd e compara. A ResNet-18 termina em `avgpool` → `fc`, então não há nada a
treinar nem hook a instalar: os $w_k^c$ são os pesos que o treino já aprendeu.

**A oclusão entra, mas com outro papel.** Um mapa de saliência é uma *afirmação*
sobre a decisão do modelo, e afirmação se testa. Para cada imagem da galeria
apagamos a região que o mapa aponta (os 10% de pixels mais quentes) e medimos o que
acontece com a probabilidade da classe prevista. Três números, e o terceiro é o que
faz o teste valer:

1. apagando a **região apontada**;
2. apagando uma região **fria de área igual** — o controle. Sem ele, qualquer queda
   seria confundida com "apagar 10% da imagem atrapalha";
3. mantendo **só** a região apontada.

O mapa só é declarado fiel quando (1) custa mais que (2). O preenchimento é zero no
espaço normalizado, que corresponde ao pixel médio do ImageNet — preto seria injetar
um padrão de alto contraste.

**O desempate das máscaras não é detalhe.** Um mapa normalizado tem milhares de
pixels exatamente iguais (zero fora da evidência), então o corte dos 10% quase sempre
cai dentro de um empate grande. Resolvê-lo na ordem do array escolheria sempre a
faixa superior da imagem — que em câmera trap é céu ou folhagem, fácil de apagar sem
custo. O controle sairia artificialmente fraco e o teste de fidelidade,
artificialmente generoso. O empate é resolvido pela **distância ao pico**: a região
quente puxa para perto, a fria para longe.

**Resolução, e o que o mapa não sustenta.** Para entrada 224×224 a `layer4` devolve
uma grade 7×7: cada célula cobre 32×32 pixels, e a ampliação é interpolação, não
informação nova. O mapa sustenta "a evidência está no canto, não no centro"; **não**
sustenta "o mapa está na pata do animal".

### A escolha das imagens é uma regra, não um gosto

Escolher a dedo "os erros mais interessantes" depois de ver os mapas seria selecionar
a evidência que confirma a história. As regras são determinísticas e saem das
probabilidades:

- **3 erros** — um por par de confusão `(verdadeira, prevista)`, na ordem de
  frequência do par, e dentro do par o erro mais confiante. Os três erros mais
  confiantes do conjunto tenderiam a ser o mesmo par repetido: a galeria mostraria um
  modo de falha três vezes em vez de três modos de falha.
- **5 acertos** — os mais confiantes, de classes distintas. O caso perigoso é o
  acerto confiante obtido pelo motivo errado, porque ele some quando a câmera muda.
  Classes distintas porque o `test_public` tem 82 imagens de urso e 4 de cavalo, e os
  cinco mais confiantes seriam cinco ursos.

Essa amostra é, de propósito, o melhor caso, e **não** é representativa. A
representatividade vem da comparação sobre as 811 imagens.

### O diagnóstico é montado de medidas, não de uma descrição da foto

O `CLAUDE.md` proíbe usar modelo fechado para classificar imagem do conjunto, e
"o mapa está no tronco da árvore ao fundo" é um ato de classificação. Então cada
etiqueta do diagnóstico vem de uma comparação mensurável — noite pelo proxy de
saturação, atenção difusa pela concentração da massa, borda pela massa na faixa
externa, localização pelo IoU entre o mapa da classe prevista e o da verdadeira,
fidelidade pela oclusão. A leitura visual fica para a apresentação, com a figura e o
gabarito do pacote na mão.

**Os limiares saem da distribuição do próprio conjunto**, não de constantes
escolhidas a mão: "difuso" significa *abaixo da mediana das 811 imagens*, "na borda"
significa *acima do percentil 75*. Duas exceções declaradas (`LOW_IOU` e
`MIN_BORDER_EXCESS`) existem porque o percentil sozinho não protege contra diferença
irrelevante — a faixa de borda ocupa 64% da área, e num conjunto de mapas difusos o
percentil 75 cai rente a esse valor, fazendo a etiqueta disparar com 0,03 ponto de
excesso. As duas aparecem na tabela com o número cru ao lado.

**Não existe etiqueta "animal pequeno".** Medir o tamanho do animal exigiria a caixa
dele, e o pacote não traz anotação de detecção. O substituto honesto é "região usada
pequena", que fala do mapa e não do animal.

### A pergunta inversa é respondida nas 811, não nas 5

O enunciado pergunta se o modelo acertou olhando para o animal ou para o fundo. Cinco
imagens ilustram; elas não respondem. As medidas do mapa são comparadas entre acertos
e erros no `test_public` inteiro, **e sempre controlando a classe** — porque os erros
se concentram nas classes difíceis, e a média bruta pode estar medindo "classes
diferentes têm mapas diferentes". É o mesmo golpe que a D19 levou com a hipótese da
noite, em que o sinal inverteu ao controlar a classe.

Com isso, `compare_groups` da trilha C passou a delegar para
`src/analysis/groups.py:compare_measure`, genérico: acurácia é a medida 0/1
"acertou". Uma implementação do controle de classe, dois usos. Pelo mesmo motivo o
proxy de noite saiu de `src/tracks/c/analysis.py` para `src/analysis/images.py`: a
Parte 3 não pertence a trilha nenhuma, e apagar a trilha C não pode levá-la embora
(D18).

**Limitação que vale dizer na apresentação.** Nenhuma destas medidas diz se o que
está **sob** o mapa é um animal. Elas dizem **onde** o modelo se baseou. Distinguir
uma coisa da outra exigiria a caixa do animal, que o pacote não tem.

---

## Em aberto

- **Limiar por classe.** A correção de viés alinhada ao diagnóstico que medimos
  (D17), não à previsão do enunciado. Fora do prazo de 16/11.
- **Hiperparâmetros da Parte 0.** lr, batch e épocas são valores padrão sem
  respaldo nas aulas disponíveis; se forem ajustados, é no `val`.
