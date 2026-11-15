# Status

Atualizado em 14/11/2025 (quarta revisão do dia).

## Feito

- **Estrutura do pacote** (`src/`): configuração, dados, modelos, estratégias,
  treino, avaliação, métricas, análise e rastreamento. Decisões em
  `docs/decisions.md` (D1–D21); padrões documentados no `README.md`.
- **Barreira do `test_public`**: toda função de seleção recusa conjunto que não seja
  o `val`, com teste de contrato que cobre funções novas por introspecção.
- **Parte 0 completa**, 3 seeds por versão, orçamento de 5 rótulos.
- **Parte 1 completa** — Trilha B (FixMatch), com os dois diagnósticos obrigatórios.
- **Trilha C completa** — professor medido e os três tratamentos de ruído treinados.
  É o **método entregue**: 0,526 de F1 macro, o melhor resultado do projeto.
- **Trilha B mantida como registro de aprendizado** — ela é o braço de controle que
  explica por que a Trilha C funcionou (D20). Não é o método entregue.
- **Arquitetura por trilha** (`src/tracks/b/`, `src/tracks/c/`): cada trilha é
  removível apagando a pasta, com a independência verificada por teste (D18).
- **Parte 2 completa** — curva de orçamento nos três orçamentos, com a reta do
  professor.
- **Parte 3 implementada** — CAM, galeria e diagnóstico medido; falta rodar
  (`make parte3`, ~1 min).
- **343 testes passando** (`make test`), `ruff` limpo (`make fmt`).
- `docs/aulas_index.md` — índice das aulas, com o que **não** está nelas.

### Números da Parte 0 (`test_public`, média ± desvio sobre 3 seeds)

| Versão | Parâmetros | F1 macro | Acurácia |
|---|---:|---|---|
| (a) `small_cnn` do zero | 394.293 | **0,234 ± 0,028** | 0,268 ± 0,025 |
| (b) ResNet-18 ImageNet + fine-tuning | 11.187.285 | **0,465 ± 0,038** | 0,489 ± 0,028 |
| teto de referência (monitor, todos os rótulos) | — | 0,686 | — |

O ImageNet recupera **51% da distância** entre o piso do zero e o teto.

Artefatos: `outputs/p0/tabela.md`, `f1_por_classe.csv`, `confusao_finetune.png`,
`curva_finetune.png`.

### Números da Parte 1 (`test_public`, orçamento de 5)

| Método | Seeds | F1 macro | Acurácia | vs. piso |
|---|---:|---|---|---|
| `piso_finetune` | 3 | 0,465 ± 0,038 | 0,489 ± 0,028 | — |
| `piso_aug_weak` (controle) | 1 | **0,482** | 0,502 | +0,017 |
| `fixmatch` | 3 | 0,438 ± 0,005 | 0,490 ± 0,025 | −0,027 |
| `fixmatch_da` | 3 | 0,438 ± 0,033 | 0,481 ± 0,043 | −0,027 |

**Resultado negativo em F1, positivo no diagnóstico** (ver D17). O FixMatch
corrigiu o colapso de classe do pseudo-rotulador — classes mudas de **7 para 2**,
cobertura de 22,2% para 41,7% em τ=0,80 — ao preço de precisão (95,6% → 86,1%). O
viés de classe medido é **por confiança, não por frequência**, o que contraria a
previsão do enunciado: a correlação entre suporte real e razão de viés é de +0,125.

Artefatos: `outputs/p1/tabela.md`, `ganho_por_classe.csv`, `vies_classes.csv`,
`pseudo_rotulos_tau_*.png`, `vies_classes.png`, `curva_tau_*.json`.

### Números da Trilha C — o professor (`test_public`, zero-shot)

| Professor | Pré-treino | F1 macro | Acurácia |
|---|---|---|---|
| `clip` | ViT-B/16, imagem-texto da web | **0,350** | 0,379 |
| `bioclip` | ViT-B/16, TreeOfLife-10M (biologia) | **0,313** | 0,363 |

**Os dois são piores que o piso (0,465)**, e o especializado em biologia é pior que o
genérico — mesma arquitetura, então a diferença mede o pré-treino. O `bioclip` é
**confiantemente errado** (confiança 0,61 ao errar, separação de apenas +0,225 contra
+0,352 do `clip`), e por isso é o `clip` que rotula o `train` nos configs do aluno.

Em τ = 0,80 o `clip` mantém 20,5% das imagens a 92,2% de precisão, mas deixa só
**13 das 21 classes** vivas — a mesma patologia da D17, num pseudo-rotulador de
origem completamente diferente. Ver D19.

Artefatos: `outputs/teacher/zero_shot.md` e os quatro `.npz` (professor × conjunto),
que são o insumo dos alunos e **não precisam ser regerados**.

### Números da Trilha C — os alunos (`test_public`, orçamento de 5, 1 seed)

| Método | F1 macro | `val` |
|---|---:|---:|
| `aluno_sem_tratamento` | 0,312 | 0,3587 |
| `aluno_rotulos_suaves` | 0,351 | 0,4602 |
| **`aluno_filtro_confianca`** | **0,526** | **0,4675** |
| piso | 0,465 ± 0,038 | 0,3913 |
| teto | 0,686 | — |

**Melhor resultado do projeto**, recuperando 28% da distância entre piso e teto. E a
minha previsão da D19 foi **falsificada**: eu havia registrado que o filtro perderia
dos rótulos suaves, e ele ganhou por +0,175. A lição corrigida (D20): *um filtro de
confiança funciona quando a confiança vem de uma fonte externa e congelada, e falha
quando vem das predições em evolução do próprio modelo*. Só visível por termos as
duas trilhas.

O aluno supera o professor em **14 de 21 classes**, e ganha mais onde o rótulo do
professor é mais sujo (Pearson −0,608 entre pureza e ganho). `bird` colapsou para
F1 = 0 — a patologia que eu previ, em uma classe.

### Números da Parte 2 (`test_public`, F1 macro)

| Orçamento | Rótulos | `piso_finetune` | `aluno_filtro_confianca` | Vantagem do aluno |
|---:|---:|---|---|---:|
| 1 | 21 | 0,260 (1 seed) | **0,432** (1 seed) | +0,172 |
| 5 | 105 | 0,465 ± 0,038 | **0,526** (1 seed) | +0,061 |
| 20 | 420 | 0,637 (1 seed) | **0,659** (1 seed) | +0,022 |
| todos | 6.505 | teto de referência: 0,686 | — | — |

**A vantagem do método encolhe conforme o orçamento cresce**, que é exatamente o que
se espera de um método que compensa falta de rótulo: com 420 rótulos o piso já faz
0,637 e sobra pouco para o professor acrescentar. Com 21 rótulos, o aluno faz com
**21 imagens rotuladas** o que o piso só faz com cerca de 105.

Com 420 rótulos o aluno chega a **0,659 contra o teto de 0,686** — a 0,027 de um
modelo treinado com os 6.505 rótulos, usando 6,5% deles.

A tabela do `outputs/p2/tabela.md` traz a ressalva dos passos por época, que aponta
para os dois lados (ver `README.md`, Parte 2): no orçamento de 1, o aluno passa
~44 vezes pelos 21 rótulos **por época**, então memorização é a primeira suspeita
para qualquer estranheza naquele ponto.

Artefatos: `outputs/p2/curva_orcamento.png`, `outputs/p2/tabela.md`,
`curva_orcamento.csv`.

### Inferência

`predict.py` classifica as 811 imagens do `test_public` em **15,6 s** em CPU. O
limite do enunciado é 10 min para ~800 imagens — margem de ~38×.

## Em andamento

Nada roda no momento. A Parte 3 está pronta para rodar (`make parte3`, ~1 min).

**Regra aprendida na prática:** nunca rodar dois trabalhos de GPU ao mesmo tempo
nesta placa de 8,6 GB. Um relatório rodado na GPU durante um treino derrubou uma
execução inteira com `hipErrorLaunchFailure`.

## Próximo passo

Nenhum item abaixo precisa de implementação nova, exceto o 2.

1. **Parte 3 — rodar** (`make parte3`, **~1 min de GPU**; `make parte3-cpu` roda em
   ~4 min sem tocar na placa). Produz `outputs/p3/`: os 3 erros e os 5 acertos com
   CAM, o diagnóstico medido de cada um, e a pergunta inversa respondida nas 811
   imagens com a classe controlada. Validado de ponta a ponta num pacote de dados
   sintético; o que falta é rodá-lo nos dados reais.
2. **Modelo entregue — muda para a Trilha C.** O melhor no `val` é
   `p1c_filter_b5_s0` (**0,4675**), contra 0,4103 do `p0_finetune_weak`. É
   entregável: o aluno é a ResNet-18 de 11.187.285 parâmetros, carrega pesos locais,
   e o professor não participa da inferência. `configs/entrega.yaml` e
   `weights/modelo.pt`, que o `predict.py` espera por padrão, **ainda não existem** —
   precisa de um `scripts/entrega.py` para materializá-los a partir da seleção no
   `val`. **É o único item que ainda exige código novo**, e é o que faz a interface
   obrigatória do enunciado funcionar.
3. **M5**: pesos, `AI_LOG.md` (falta o episódio da Trilha C e da previsão
   falsificada; o da Parte 3 já está escrito), teste em ambiente limpo.

## Riscos

| Risco | Impacto | Mitigação |
|---|---|---|
| O `predict.py` não roda com os padrões: faltam `configs/entrega.yaml` e `weights/modelo.pt` | a interface obrigatória do enunciado não funciona em máquina limpa | é o único item que ainda precisa de código (`scripts/entrega.py`), e não precisa de treino: os pesos já existem em `outputs/p1c_filter_b5_s0/best.pt` |
| Duas trilhas em vez de uma | dobra os diagnósticos obrigatórios a apresentar | a C é o método entregue e a B é o controle que explica por que ela funcionou (D20). Se faltar tempo de **apresentação**, a B encolhe para 2-3 slides; ela não sai do repositório |
| A trilha entregue é a que não está nas aulas | se a banca puxar CLIP ou zero-shot, não há aula para apoiar | a seção de Visual FM do `A2.pdf` do aluno está vazia e as aulas não cobrem CLIP. A Trilha B, que **está** no material, serve de terreno firme para a parte conceitual |
| `bird` colapsou para F1 = 0 no modelo entregue | uma das 21 classes nunca é prevista | medido e reportado em `outputs/p1c/aluno_vs_professor.csv`; é consequência direta do filtro, e a explicação está na D20 |
| RAM estourou numa sessão anterior | perde execução no meio | `num_workers: 4` nos configs da Trilha C e `--set data.num_workers=2` documentado no README; **nunca** rodar dois trabalhos de GPU ao mesmo tempo nesta placa |
| Época escolhida no `val` é instável | a "melhor época" é ruído | 13 épocas ficam a menos de 0,01 do melhor (seed 0); melhores épocas por seed: 24, 18, 54. Tratar como platô, não como escolha |
| `horse` tem 4 imagens no `test_public` | o F1 de cavalo oscila com um acerto a mais ou a menos | sempre reportar por classe, com o suporte. O enunciado diz que isso faz parte do problema |
| Controle de aumentação tem 1 seed | o +0,017 pode ser ruído de seed | está marcado como 1 seed em toda tabela; se sobrar GPU, rodar as outras 2 seeds antes de qualquer coisa opcional |
| τ escolhido com a lr antiga | o τ pode não ser o ótimo para lr = 3e-5 | registrado em D16 como limitação; refazer a varredura custaria 40 min que concorrem com a Parte 2 |

## Bloqueios

- **Aprovação para criar `scripts/entrega.py`**, que materializa `configs/entrega.yaml`
  e `weights/modelo.pt` a partir da seleção no `val`. Sem ele o `predict.py` não roda
  com os padrões, e essa interface é obrigatória no enunciado.
- Nada mais está bloqueado: Partes 0 a 3 são `make`.

## Marcos

| Marco | Entrega | Meta | Situação |
|---|---|---|---|
| M0 | Índice das aulas, estrutura, dados, testes e `predict.py` | 09/11 | **feito** em 12/11 |
| M1 | Parte 0: piso, curvas e matriz de confusão | 10/11 | **feito** em 12/11 |
| M2 | Parte 1: trilha escolhida e experimentos obrigatórios | 13/11 | **feito** em 14/11 — Trilha C é a entregue (0,526); Trilha B fica como registro de aprendizado |
| M3 | Parte 2: curva de orçamento e ganho por classe | 14/11 | **feito** em 14/11 — o aluno vence o piso nos três orçamentos |
| M4 | Parte 3: falhas, acertos e saliência | 15/11 | **implementado** em 14/11, validado em dados sintéticos; falta rodar (`make parte3`) |
| M5 | Ambiente limpo, README, pesos e `AI_LOG.md` | 16/11 12h | não iniciado |

**Se o prazo apertar**, cortar nesta ordem: o detalhamento da Trilha B na
apresentação (ela fica como registro no repositório), depois a comparação agregada da
Parte 3 (`atalho.md` e `distribuicoes.png`, que são extras — o obrigatório são os 3
erros e os 5 acertos). **A Trilha C não é mais candidata a corte** — ela é o
método entregue. Nunca cortar: os experimentos obrigatórios da trilha, a curva de
orçamento e o `predict.py`.
