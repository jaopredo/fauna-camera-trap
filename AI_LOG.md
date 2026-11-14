# Uso de IA neste trabalho

Registro dos episódios em que recorremos a ferramentas de IA, o que validamos e o
que mudamos. A política do enunciado (seção 5) permite o uso; o que não se admite
é entregar algo que não entendemos.

Nenhuma imagem do conjunto foi mostrada a um modelo fechado para classificação ou
rotulagem, em nenhum momento.

---

## Episódio 1 — Estrutura inicial do repositório (08/11)

**Problema.** Montar a organização do projeto antes de decidir qualquer modelo,
evitando o retrabalho de reorganizar tudo quando a trilha fosse escolhida.

**Como a IA foi usada.** Claude Code analisou um projeto nosso anterior
(`semantic-segmentation`), que serviu de referência de padrão, e o enunciado do
PA3, para propor a divisão de módulos. Pedimos que ele questionasse as escolhas do
projeto antigo em vez de copiá-las.

**O que validamos.** Três pontos em que a proposta contrariou a referência e nós
conferimos antes de aceitar:

1. O `AppConfig` Singleton do projeto antigo devolve a configuração da primeira
   chamada quando um segundo YAML é carregado. Como a Parte 2 roda 3 seeds × 3
   orçamentos, isso faria as seeds rodarem iguais sem aviso. Conferimos abrindo
   `src/core/config.py` do projeto antigo: o `_instances` da metaclass realmente
   ignora os argumentos. Trocamos por função que devolve dataclass congelada.
2. A proposta inicial dele era separar validação e teste em módulos paralelos,
   como pedimos. Verificamos que no projeto antigo `training/config.py` e
   `evaluation/config.py` são arquivos idênticos, e mudamos para um único
   avaliador parametrizado pelo conjunto, com a barreira do `test_public`
   aplicada nas funções de seleção.
3. A contagem de parâmetros e o teto de 12M foram conferidos contra a fórmula
   exata do enunciado, com teste que falha num `nn.Linear(4000, 4000)`.

**O que mudamos.** Além dos itens acima: `model` e `strategy` passaram a ser seções
obrigatórias na configuração (a IA havia colocado um valor padrão vazio, que
mentia sobre haver um backbone escolhido), e o estado da estratégia de treino ganhou
campo próprio no checkpoint — a primeira versão restaurava a configuração do
experimento no lugar do estado da estratégia ao retomar um treino.

Depois pedimos que os nomes das classes seguissem o padrão de projeto que cada uma
implementa (`TrainingStrategy`, `ModelFactory`, `TrainingObserver`), para que o
nome dissesse onde uma implementação nova entra. Está registrado em
`docs/decisions.md` (D7).

---

## Episódio 2 — Parte 0: os dois modelos do piso (12/11)

**Problema.** Implementar as duas versões do piso (CNN do zero e backbone ImageNet
com fine-tuning), escolher o backbone que valeria para o resto do PA, e reportar as
métricas que o enunciado pede.

**Como a IA foi usada.** Claude Code indexou os PDFs das aulas por página para
localizar o que havia sobre fine-tuning e backbone, propôs três backbones com
contagem de parâmetros e trade-offs, implementou as estratégias e rodou as 6
execuções (3 seeds × 2 versões).

**O que validamos.**

1. **A contagem de parâmetros.** A IA estimou 11,19 M para a ResNet-18 com 21
   classes antes de poder executar (o `torchvision` ainda não estava instalado) e
   marcou isso explicitamente como não verificado. Depois da instalação, o valor
   medido foi **11.187.285** — 93,2% do teto. O número está travado em
   `tests/test_models.py`, de modo que qualquer mudança futura falha antes do treino.
2. **O que as aulas realmente dizem.** Pedimos que ela não citasse aula que não
   tivesse lido. O resultado está em `docs/aulas_index.md`: fine-tuning aparece na
   aula 15 p. 15 e a ResNet-18 como backbone na aula 13 p. 60, mas **transfer
   learning supervisionado, projeto de CNN e aumentação supervisionada não estão**
   nas aulas 11–15 que temos. As decisões que dependem disso (lr, otimizador, batch,
   épocas) estão marcadas como "fora do material da disciplina" em `decisions.md`.
3. **A "melhor época" é ruído.** A IA inicialmente relatou que o melhor F1 de
   validação da seed 0 estava na época 8. Ao fechar as 60 épocas, o argmax tinha ido
   para a época 54, com 0,3841 contra 0,3820 — diferença de 0,002. Conferimos: 13
   épocas ficam a menos de 0,01 do melhor, e as melhores épocas das três seeds são
   54, 38 e 18. Ou seja, o `val` de 202 imagens não distingue essas épocas. O que a
   curva mostra de verdade é um platô a partir da época 8, e é assim que
   apresentamos.

**O que mudamos.** Dois defeitos que só o uso real expôs: o `labeled_loader` usava
`drop_last=True` no treino, o que descartava até 9 das 105 imagens rotuladas por
época; e o `Makefile` chamava o `python3` do sistema, sem as dependências, então
`make test` quebrava na coleta. Também acrescentamos `flush` ao log por época, porque
em execução redirecionada a saída ficava presa no buffer até o fim do treino.
