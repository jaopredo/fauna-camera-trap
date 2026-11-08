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
