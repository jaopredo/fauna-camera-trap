# PA3 — Poucos rótulos: fauna em câmeras trap

Classificação de 21 espécies do ENA24 com 5 rótulos por classe e 6.505 imagens de
treino sem rótulo. O produto final é um modelo de no máximo 12 milhões de
parâmetros que roda em CPU, sem internet.

## Ambiente

Python 3.12 (a versão do Colab). Com [uv](https://docs.astral.sh/uv/):

```bash
make setup          # dependências + torch detectando a placa
make ambiente       # confere qual roda de torch ficou instalada
```

**`torch` e `torchvision` não estão no `pyproject.toml`, de propósito.** A roda certa
depende da placa, e fixá-la nas dependências obrigaria todo mundo ao mesmo backend.
`make setup` instala os dois à parte com `--torch-backend=auto`, que detecta o
hardware: resolve para `+rocmX.Y` numa Radeon, `+cuXXX` numa NVIDIA e CPU se não
houver GPU.

```bash
make setup-cpu      # força a roda de CPU (Colab sem GPU, CI, ou medir a inferência)
make setup-pip      # sem uv: venv + pip + requirements.txt
make setup-cpu-pip  # idem, forçando CPU
```

O `requirements.txt` versionado também não traz torch, pelo mesmo motivo — os alvos
`setup-pip` o instalam em seguida.

**Só para a Trilha C**, o professor zero-shot precisa de uma dependência a mais:

```bash
make setup-professor   # open_clip_torch + os pesos abertos (BioCLIP, CLIP)
```

Ela fica fora do `pyproject.toml` pelo mesmo motivo do torch: `open_clip` **depende**
de torch, e deixá-la no lock faria o `uv sync` instalar a roda padrão por cima da sua.
Instalada depois do torch, o uv vê a dependência satisfeita e não toca na roda
ROCm/CUDA — o alvo confere isso e avisa se tiver mexido. Não entra no `make setup`
porque quem apagar a Trilha C não precisa dela.

### Memória

`num_workers` são processos carregando imagens, cada um com buffer próprio: bom numa
máquina folgada, pesado numa apertada. Os configs da Trilha B usam 8 (foi com eles que
os resultados registrados saíram, então ficaram como estão); os da Trilha C usam 4.
Se faltar RAM, baixe sem editar o config:

```bash
.venv/bin/python scripts/train.py --config configs/p1c_distill_soft.yaml \
    --set data.num_workers=2
```

`num_workers` não afeta resultado nenhum — só a velocidade de carregamento. E **não
rode dois treinos ao mesmo tempo**, nem um relatório na GPU durante um treino: numa
placa de 8 GB os dois contextos competem e o ROCm derruba um deles com
`hipErrorLaunchFailure`. Aconteceu aqui e custou uma execução pela metade.

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
python scripts/train.py --config configs/p0_finetune.yaml
```

O YAML declara quantas seeds o experimento tem (`seeds: [0, 42, 123]`), e o comando
acima roda **todas** elas, cada uma no seu diretório. Variações sem criar outro YAML:

```bash
python scripts/train.py --config configs/p0_finetune.yaml --seed 42          # só uma delas
python scripts/train.py --config configs/p0_finetune.yaml --set data.budget=20
python scripts/train.py --config configs/p0_finetune.yaml --set seeds=[0,7]
python scripts/train.py --config configs/p0_finetune.yaml --resume
```

`configs/exemplo.yaml` existe só para documentar o schema: os nomes de `model`,
`strategy` e `augment` nele são `TBD`, e rodá-lo falha de propósito, listando o que
está registrado.

Cada execução grava em `outputs/<nome>_b<orçamento>_s<seed>/`: `manifest.json`
(configuração, seed e commit), `history.json`, `best.pt` e `last.pt`.

## Um comando que avalia

```bash
python scripts/evaluate.py --config configs/p0_finetune.yaml \
    --checkpoint outputs/p0_finetune_b5_s0/best.pt --split val
```

`--split val` é o conjunto de seleção de modelo; `--split test_public` só reporta
números. A regra está no código, não só na documentação: as funções de seleção em
`src/evaluation/selection.py` recusam qualquer resultado que não venha do
`val`.

## Reproduzir a Parte 0 — o piso

Duas versões supervisionadas com os 105 rótulos de `rotulos_5.csv`: **(a)** uma CNN
pequena do zero (`small_cnn`, 394.293 parâmetros) e **(b)** ResNet-18 do ImageNet com
fine-tuning (11.187.285 parâmetros, 93% do teto de 12 M).

```bash
make parte0
```

Equivalente, sem o Makefile:

```bash
python scripts/train.py --config configs/p0_scratch.yaml
python scripts/train.py --config configs/p0_finetune.yaml
python scripts/report_p0.py
```

As três seeds estão declaradas nos próprios YAMLs (`seeds: [0, 42, 123]`), que é onde
o protocolo do enunciado — 3 seeds no orçamento de 5, uma nos de 1 e 20 — fica
registrado. Nenhum laço de shell prescreve seed.

O relatório escreve em `outputs/p0/`: `tabela.md` (F1 macro e acurácia no
`test_public`, média ± desvio sobre as 3 seeds), `f1_por_classe.csv`,
`confusao_finetune.png` e `curva_finetune.png`; e acrescenta as linhas do orçamento
de 5 em `curva_orcamento.csv`.

As figuras saem de uma seed fixada de antemão (`--figuras-seed`; sem ela, a primeira
declarada em `seeds`), nunca da melhor seed no `test_public` — escolher pelo teste
seria seleção de modelo nele.
 
## Reproduzir a Parte 1 — Trilha B (FixMatch)

FixMatch com os 105 rótulos e as 6.505 imagens do `train` sem rótulo, no mesmo
backbone da Parte 0.

```bash
make parte1-controle   # piso com a aumentação fraca — controle do confundidor
make parte1-sweep      # varre τ ∈ {0,70; 0,80} e escolhe no val
make parte1            # 3 seeds × (FixMatch, FixMatch+DA) + relatório
```

`make parte1-sweep` termina chamando `report_p1.py --sweep`, que compara as execuções
**no `val`** via `best_candidate` — função que levanta exceção se receber resultado do
`test_public`. As curvas de precisão e cobertura saem do `test_public`, como o
enunciado manda, e não alimentam escolha nenhuma.

O relatório escreve em `outputs/p1/`: `tabela.md` (piso, FixMatch e FixMatch+DA, com
o ganho sobre o piso), `ganho_por_classe.csv`, `pseudo_rotulos_tau_*.png` (precisão e
cobertura por classe contra τ), `vies_classes.png` (para onde vão os pseudo-rótulos) e
`vies_classes.csv` (os números por trás da figura, com a razão de viés por classe).

**O resultado é negativo em F1 e positivo no diagnóstico.** O FixMatch fica abaixo do
piso (0,438 contra 0,465) e abaixo do controle de aumentação (0,482), mas corrigiu o
colapso de classe do pseudo-rotulador: classes sem nenhum pseudo-rótulo caíram de 7
para 2 e a cobertura em τ = 0,80 subiu de 22,2% para 41,7%, ao preço de precisão
(95,6% → 86,1%). E o viés medido é **por confiança, não por frequência**, o que
contraria a previsão do enunciado. Ver `docs/decisions.md`, D17.

**Por que τ = 0,80 e não o 0,95 do paper:** medido no `val`, τ = 0,95 deixa 6 das 21
classes sem nenhum pseudo-rótulo. Ele foi calibrado em CIFAR-10, com 10 classes
balanceadas. Ver `docs/decisions.md`, D12.

## Reproduzir a Parte 1 — Trilha C (professor e aluno)

Um foundation model de pesos abertos rotula as 6.505 imagens do `train`, e uma
ResNet-18 (o aluno) aprende desses pseudo-rótulos junto com os 105 rótulos
verdadeiros. **Independente da Trilha B**: nenhum comando abaixo depende de nada da
trilha anterior.

```bash
make setup-professor    # uma vez: open_clip + pesos abertos
make parte1c            # professor + as três variantes + relatório
```

Ou passo a passo:

```bash
make parte1c-professor  # roda o professor e grava outputs/teacher/*.npz  (~10 min de GPU)
make parte1c-sweep      # os três tratamentos de ruído, 1 seed cada       (~35 min de GPU)
make parte1c-relatorio  # só o relatório, das execuções que já existem
```

### Como funciona

O professor roda **uma vez** e grava as probabilidades por imagem num `.npz`. Depois
disso ele sai de cena: o aluno lê o arquivo e nunca carrega um foundation model. Essa
fronteira é o que torna *"o professor não pode fazer parte do modelo entregue"*
**estrutural** em vez de disciplina — `predict.py` não tem caminho até ele.

### Integridade experimental

- **O professor nunca vê o `val`.** `scripts/teacher.py --splits val` é recusado com
  `TeacherPermissionError`, não com um comentário. Um teste por introspecção enumera
  as funções públicas do módulo e exige que toda função que receba um conjunto passe
  pela barreira, para que uma função nova não abra um buraco silencioso.
- **O template do prompt é fixo**, `"a photo of a {}."`, declarado no módulo e não na
  linha de comando. Escolher entre templates seria seleção de modelo, e os dois
  conjuntos disponíveis para isso estão fechados: o `val` o professor não pode ver, e
  o `test_public` é proibido para escolha. Sem escolha, não há regra a interpretar.
- **O `test_public` é usado para medir o professor**, que é o que o enunciado manda
  explicitamente ("para vocês medirem o professor"). Ele não alimenta escolha nenhuma.

### Os três tratamentos de ruído (obrigatório 3)

| Config | Tratamento | O que testa |
|---|---|---|
| `p1c_distill_hard.yaml` | nenhum | quanto ruído o professor injeta sem tratamento |
| `p1c_distill_filter.yaml` | filtro por confiança, τ = 0,80 | a escolha ingênua |
| `p1c_distill_soft.yaml` | rótulos suaves, T = 2,0 | usa as 6.505, sem filtrar classe |

**Há uma previsão sendo testada aqui.** A Trilha B mediu que um limiar de confiança
não filtra *erro*, filtra *incerteza* — e a incerteza se concentra nas classes que o
modelo confunde, então o limiar vira um filtro de classe disfarçado e o F1 macro pune
isso (ver `docs/decisions.md`, D17). Se a previsão valer, `filter` perde de `soft`
apesar de treinar com rótulos mais limpos. A previsão foi registrada **antes** de
rodar os alunos.

### O que já foi medido: o professor

F1 macro no `test_public`, zero-shot, com os nomes das classes como prompt
(`outputs/teacher/zero_shot.md`):

| Professor | Pré-treino | F1 macro | Acurácia |
|---|---|---|---|
| `clip` | ViT-B/16, pares imagem-texto da web | **0,350** | 0,379 |
| `bioclip` | ViT-B/16, TreeOfLife-10M (biologia) | **0,313** | 0,363 |
| nosso piso supervisionado (105 rótulos) | — | 0,465 | 0,489 |

**Os dois professores são piores que o piso**, e o especializado em biologia é pior
que o genérico. Os dois são a mesma arquitetura, então a diferença mede o pré-treino,
não o tamanho.

Calibração, que decide se filtrar por confiança pode funcionar:

| Professor | Confiança quando acerta | Quando erra | Separação |
|---|---|---|---|
| `clip` | 0,710 | 0,358 | **+0,352** |
| `bioclip` | 0,833 | 0,608 | +0,225 |

O `clip` é razoavelmente calibrado; o `bioclip` é **confiantemente errado**. Em
τ = 0,80 o `clip` mantém 20,5% das imagens com 92,2% de precisão, mas deixa só
**13 das 21 classes** vivas — o colapso que a Trilha B previu, visível antes de
treinar aluno nenhum.

> **Atenção ao `ViT-B-16-quickgelu`.** Os pesos da OpenAI foram treinados com
> QuickGELU, e o `open_clip` tem arquiteturas separadas. Com o nome sem sufixo ele
> carrega os pesos numa rede de GELU padrão, **avisa e roda**, produzindo um F1
> silenciosamente diferente (medimos 0,384 com a ativação errada antes de notar o
> aviso). Está travado em `tests/test_track_c_teacher.py`.

### O que foi medido: os alunos

| Tratamento | F1 macro `test_public` | F1 macro `val` |
|---|---:|---:|
| `aluno_sem_tratamento` | 0,312 | 0,3587 |
| `aluno_rotulos_suaves` | 0,351 | 0,4602 |
| **`aluno_filtro_confianca`** | **0,526** | **0,4675** |
| piso | 0,465 ± 0,038 | 0,3913 |
| teto | 0,686 | — |

**É o melhor resultado do projeto** e o método entregue, escolhido no `val`. Recupera
28% da distância entre o piso e o teto.

**A previsão registrada antes de rodar foi falsificada.** A `docs/decisions.md` D19
dizia que o filtro perderia dos rótulos suaves, porque a Trilha B havia mostrado que
um limiar de confiança colapsa classes. O filtro ganhou por +0,175. A lição corrigida
na D20:

> um filtro de confiança funciona quando a confiança vem de uma fonte **externa e
> congelada**; falha quando vem das predições **em evolução do próprio modelo**

Na Trilha B o pseudo-rotulador era o próprio modelo, e filtrar pela própria confiança
criava realimentação. Aqui o professor é congelado — a confiança dele não co-evolui
com o aluno. O `kept_rate` medido foi **0,225**: o aluno treinou em ~1.460 das 6.505
imagens, aquelas em que o professor acerta 92,2% em vez de 35%.

**Obrigatório 3.** O aluno supera o professor em **14 de 21 classes**, e a correlação
de Pearson entre a pureza do pseudo-rótulo e o ganho do aluno é **−0,608**: ele ganha
mais onde o professor é pior. `eastern_chipmunk` é o caso extremo — professor 0,000,
aluno 0,967. E `bird` colapsou para F1 = 0, que é o preço do filtro.

### Artefatos

`outputs/teacher/zero_shot.md` (obrigatório 1) e, em `outputs/p1c/`: `tabela.md`,
`ruido_<professor>.png` e `.csv` (obrigatório 2 — a matriz de ruído e as confusões
sistemáticas), `calibracao.png`, `aluno_vs_professor.png` e `.csv` (obrigatório 3) e
`noite_vs_dia.md`.

Sobre `noite_vs_dia.md`: o enunciado sugere procurar erro sistemático em fotos
noturnas. Câmera trap noturna usa infravermelho, então a saturação média dos pixels é
um proxy direto, medido da própria imagem. **O resultado bruto engana** — a noite é
dominada por `virginia_opossum` e `american_black_bear`, e urso é a melhor classe do
professor, o que infla a acurácia noturna. O arquivo traz as duas leituras e marca
com ⚠ quando elas apontam para lados opostos. Já *"animais pequenos"* não é
mensurável aqui: exigiria a caixa do animal, que o pacote não traz, e afirmar sem
medir seria inventar.

## Reproduzir a Parte 2 — a curva de orçamento

Os orçamentos de 1 e 20 rótulos por classe, uma seed cada; o de 5 já saiu das Partes
0 e 1. Não há config novo: o orçamento é campo de configuração e o diretório da
execução já inclui `b<orçamento>`.

As duas curvas são o **piso** e o **método da trilha** — que é o aluno da Trilha C
com filtro por confiança, o que ganhou no `val`. O professor entra como **reta
horizontal**, porque é zero-shot e não depende do orçamento (o enunciado pede que ele
apareça na curva).

```bash
make parte2            # treina os orçamentos 1 e 20 + avalia + figura  (~24 min de GPU)
make parte2-relatorio  # só a figura e a tabela, do curva_orcamento.csv existente
make parte2-fixmatch   # opcional: os pontos de 1 e 20 da Trilha B também (~40 min)
```

`make parte2-relatorio` lê apenas o `curva_orcamento.csv` — não carrega modelo e não
toca em nenhum conjunto de dados. Escreve `outputs/p2/curva_orcamento.png` (F1 macro
contra nº de rótulos, eixo x em log, com a linha do teto) e `outputs/p2/tabela.md`.

A reta do professor não é um número digitado no script: `report_p1c.py` publica o que
mediu em `outputs/p1c/referencias.json`, e `report_p2.py` lê qualquer
`outputs/*/referencias.json` que exista. O acoplamento é **por dado, não por import**,
para que `report_p2.py` continue neutro e apagar uma trilha só faça a reta
desaparecer.

A tabela traz uma ressalva que a figura sozinha esconde, e ela aponta para **os dois
lados**:

- para o **piso**, a época é definida pelos rótulos: o orçamento de 1 recebe 20×
  menos passos que o de 20, então parte de qualquer subida vem de mais rótulos e
  parte vem de mais atualizações de peso;
- para o **aluno da Trilha C** é o contrário. A fonte primária dele é o pool de 6.505
  imagens, que não encolhe: a época tem sempre ~58 passos, e os rótulos são ciclados
  dentro dela. Com 1 rótulo por classe ele passa pelas mesmas 21 imagens **~44 vezes
  por época**; com 20, apenas ~2. **Quanto menor o orçamento, mais ele memoriza.**

Se o ponto de 1 rótulo sair ruim, essa é a primeira suspeita — e não a falta de dado
em si. É a mesma armadilha da D16.

## Reproduzir a Parte 3 — galeria de falhas e atalhos

```bash
make parte3        # ~1 min de GPU, ou `make parte3-cpu` (~4 min, sem tocar na GPU)
```

Não treina nada. Carrega o `best.pt` da execução de **melhor F1 macro de validação**
no orçamento — a mesma regra que define o modelo entregue — e produz em `outputs/p3/`:

| Arquivo | O que é |
|---|---|
| `erros.png` | os 3 erros, com o mapa da classe prevista e o da verdadeira |
| `acertos.png` | os 5 acertos, com o mapa da classe prevista |
| `galeria.md` | verdadeira, prevista, confiança, por que a imagem entrou e o diagnóstico |
| `medidas.csv` | todos os números crus de cada imagem da galeria |
| `atalho.md` | a pergunta inversa respondida nas 811 imagens, não nas 5 |
| `distribuicoes.png` | as medidas do mapa em acertos contra erros |

Nenhum nome de experimento está escrito no script: ele lista todas as execuções do
orçamento, ordena pelo `val` e usa a primeira. Trocar o modelo entregue não exige
editar nada, e a Parte 3 continua válida se uma trilha for apagada.

### CAM, e por que isso atende o "Grad-CAM" do enunciado

O mapa é **CAM** (aula 13, `DL_20242_13_Unsupervised.pdf`, p. 80–82). Numa rede que
termina em global average pooling seguido de uma camada linear — e a ResNet-18
termina assim, `avgpool` → `fc` — o peso do Grad-CAM se reduz ao peso da `fc`
dividido pelo número de posições, de modo que os dois mapas são iguais a menos de uma
constante positiva, e idênticos depois da normalização. Não é analogia:
`tests/test_saliency.py` calcula o Grad-CAM pelo autograd e compara. Consequência
prática: nada a treinar, nenhum hook.

A grade da `layer4` é 7×7 para entrada 224×224 — cada célula cobre 32×32 pixels, e a
ampliação é interpolação. O mapa sustenta "a evidência está no canto, não no centro";
**não** sustenta "o mapa está na pata do animal".

### A oclusão testa o mapa, em vez de ser um segundo mapa

Um mapa de saliência é uma afirmação sobre a decisão do modelo. Para cada imagem da
galeria, apagamos a região apontada (os 10% mais quentes, contornada em branco na
figura) e medimos a probabilidade da classe prevista — e apagamos também uma região
**fria de área igual**, como controle. Sem esse controle, qualquer queda se confunde
com "apagar 10% da imagem atrapalha". O mapa só é declarado fiel quando apagar o que
ele aponta custa mais que apagar o controle. A coluna "O mapa explica?" da
`galeria.md` traz os dois números.

### O diagnóstico é medido, e a leitura visual é sua

O `CLAUDE.md` proíbe usar modelo fechado para classificar imagem do conjunto, e
"o mapa está no tronco da árvore ao fundo" é um ato de classificação. Então cada
etiqueta vem de uma comparação mensurável: noite pelo proxy de saturação, atenção
difusa pela concentração da massa, borda pela massa na faixa externa, localização
pelo IoU entre o mapa da classe prevista e o da verdadeira, fidelidade pela oclusão.
Os limiares saem da distribuição das próprias 811 imagens ("difuso" = abaixo da
mediana), não de constantes escolhidas a mão.

**Não existe etiqueta "animal pequeno"** — medir o tamanho do animal exigiria a caixa
dele, e o pacote não traz anotação de detecção. O substituto honesto é "região usada
pequena", que fala do mapa.

A coluna IoU separa dois modos de falha que a matriz de confusão junta: **IoU alto**
é o modelo olhando para o lugar certo e nomeando errado (confusão de aparência, mais
rótulo ajuda); **IoU baixo** é o modelo olhando para outro lugar (falha de
localização, mais rótulo da mesma classe não resolve).

### A pergunta inversa é respondida nas 811, não nas 5

Cinco imagens ilustram; não respondem. O `atalho.md` compara as medidas do mapa entre
acertos e erros no `test_public` inteiro, **sempre em duas versões**: a bruta e a que
controla a classe. Os erros se concentram nas classes difíceis, então a média bruta
pode estar medindo "classes diferentes têm mapas diferentes". É o mesmo confundidor
que inverteu o sinal da hipótese da noite na D19.

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
  methods/            contrato da Strategy de treino, registry e a estratégia neutra
  training/           laço de treino e observadores (Observer)
  evaluation/         avaliação, seleção de modelo e reabertura de execuções salvas
  metrics.py          F1 macro, acurácia, matriz de confusão
  analysis/           o que não é de trilha nenhuma:
      figures.py        curvas, matriz de confusão, galeria de saliência
      budget.py         curva de orçamento da Parte 2
      saliency.py       CAM, medidas do mapa e oclusão como teste (Parte 3)
      gallery.py        quais imagens entram na galeria e o diagnóstico medido
      groups.py         comparar uma medida entre grupos controlando a classe
      images.py         medidas da imagem crua (proxy de noite)
  tracking.py         seed, commit e manifesto de cada execução
  tracks/             UMA PASTA POR TRILHA (ver abaixo)
    b/                FixMatch: estratégia, par fraco/forte, diagnósticos, figuras
    c/                professor e aluno: professor zero-shot, destilação, análises
scripts/              pontos de entrada; é onde os componentes são compostos
tests/
docs/                 decisions.md, STATUS.md e o enunciado
predict.py
```

### Uma pasta por trilha, e apagar a pasta basta

Cada trilha é autocontida em `src/tracks/<trilha>/`: a estratégia de treino, os
diagnósticos dela, as figuras dela, e nada mais. **Apagar uma trilha é apagar a pasta,
os configs e os testes dela** — sem editar nenhum arquivo compartilhado:

```bash
rm -rf src/tracks/c configs/p1c_*.yaml scripts/report_p1c.py scripts/teacher.py \
       tests/test_track_c_*.py      # a Trilha B continua intacta
```

Para isso `src/tracks/__init__.py` descobre as subpastas presentes com `pkgutil` e as
importa, para que os decoradores `@register_strategy` e `@register_augmentation`
rodem. É o **único** lugar do projeto com import implícito, e é deliberado: com um
`from src.tracks import b, c` explícito, remover uma trilha quebraria o import da
outra, que é exatamente o que se queria evitar.

Duas coisas de aparência "da Trilha B" **não** estão na pasta dela, de propósito:

- a aumentação `weak` (`src/data/transforms.py`) — o modelo entregue a usa, então ela
  é infraestrutura; apagar a Trilha B não pode levar embora a aumentação do modelo
  que entregamos. Já o par `weak_strong` é da trilha e está em `src/tracks/b/`;
- `src/analysis/budget.py` — a curva de orçamento da Parte 2 é comum às duas trilhas;
- `src/analysis/images.py` e `src/analysis/groups.py` — o proxy de noite nasceu na
  Trilha C e o controle de classe também, mas a Parte 3 usa os dois e não pertence a
  trilha nenhuma. A trilha C é quem importa deles, nunca o contrário.

**A independência é uma asserção, não uma promessa.**
`tests/test_tracks_independence.py` percorre a árvore sintática de cada módulo e
falha se uma trilha importar da outra, se a infraestrutura importar qualquer trilha,
ou se um script neutro (`train.py`, `report_p0.py`, `report_p2.py`, `report_p3.py`,
`predict.py`) depender de uma. Verificado de verdade: com `src/tracks/b/` removida, 189 testes
continuam passando.

## Padrões de projeto

Cada padrão aqui existe porque o enunciado exige **mais de uma implementação** do
mesmo papel. Onde não há variação prevista, não há padrão: `metrics.py` são funções
puras, `config/` e `tracking.py` são dataclasses congeladas.

| Padrão | Onde | Papel que varia | O que o enunciado exige |
|---|---|---|---|
| Strategy | `src/methods/` + `src/tracks/*/strategy.py` | método de treino | Parte 0 supervisionado + o método de **cada** trilha (Parte 1) |
| Strategy | `src/data/augment.py` | aumentação | experimento obrigatório da Trilha A; par fraco/forte do FixMatch |
| Factory com Registry | `src/tracks/__init__.py` | qual trilha existe | duas trilhas implementadas, cada uma removível sem editar código |
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

**Como adicionar uma trilha** — uma pasta nova, nenhuma edição em nada existente:

```python
# src/tracks/a/strategy.py
@register_strategy("simclr")
def build_simclr(*, num_classes: int, temperature: float = 0.5) -> TrainingStrategy:
    return SimCLR(num_classes=num_classes, temperature=temperature)
```

```python
# src/tracks/a/__init__.py — importa o que registra
from src.tracks.a import strategy  # noqa: F401

__all__ = ["strategy"]
```

Só isso. `src/tracks/__init__.py` descobre a pasta sozinho, o decorador roda, e
`strategy.name: simclr` passa a valer no YAML. `tests/test_strategy_contract.py`
cobre a estratégia nova **automaticamente**, por `available_strategies()`, e
`tests/test_tracks_independence.py` passa a verificar que ela não importa das outras.

Estratégias que precisam de um artefato externo para nascer — a destilação da Trilha
C precisa do `.npz` que o professor gravou — devem recusar com um `AppError` que diz
**o que falta e como obter**, não com um `TypeError` de Python: quem erra isso está
escrevendo um YAML, não chamando a função. O teste de contrato compartilhado aceita
as duas formas (cumprir o contrato, ou explicar o que falta), e o contrato em si é
verificado pelo teste da própria trilha, que sabe montar o artefato.

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
