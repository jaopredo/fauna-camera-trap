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

## Episódio 3 — Parte 1: quatro causas descartadas antes de aceitar o resultado (13–14/11)

**Problema.** Implementar FixMatch (Trilha B) e os dois diagnósticos obrigatórios. O
primeiro resultado foi ruim e o FixMatch ficou claramente abaixo do piso. A pergunta
era se havia um defeito na implementação ou se o método simplesmente não funciona com
105 rótulos e 21 classes.

**Como a IA foi usada.** Claude Code localizou a definição do FixMatch na aula 13
(p. 85–91) e implementou a partir dela, não de memória do paper; marcou o
alinhamento de distribuição como **fora do material da disciplina**, porque o
enunciado o oferece pelo nome mas ele não aparece nas aulas 11–15. Depois,
instrumentou o treino para distinguir "método ruim" de "código errado".

**O que validamos.** Quatro hipóteses, todas medidas antes de concluir qualquer
coisa:

1. **BatchNorm (defeito real, corrigido).** A primeira versão fazia dois forwards —
   um na vista fraca, um na forte. Com o modelo em modo treino, isso atualiza as
   estatísticas acumuladas **duas vezes por passo**, com distribuições diferentes, e
   o pseudo-rótulo acabava calculado sob uma normalização que não era a da vista
   forte contra a qual ele é aplicado. Corrigido com um forward único sobre
   `cat([rotulado, fraca, forte])`. Custo: 6,2 GB de VRAM e 39 s por época contra 28.
2. **O limiar do paper não serve aqui.** τ=0,95 é calibrado em CIFAR-10, com 10
   classes balanceadas. Medimos no `val`: ele deixa **6 das 21 classes** sem nenhum
   pseudo-rótulo. A grade {0,70; 0,80} foi escolhida no `val`.
3. **A taxa de aprendizado mudou de significado (o achado do trabalho).** Com o pool
   sem rótulo como fonte primária, a época passa de 7 para ~58 passos, e em 30
   épocas são ~265 passagens pelos 105 rótulos contra 60 na Parte 0. A mesma lr
   memoriza 4× mais: com 1e-4 a `ce` caiu para 0,0006, a confiança média subiu para
   0,941 e a `mask_rate` chegou a **0,995** — o limiar deixou de filtrar enquanto a
   precisão do pseudo-rótulo caía para 36,5%. O modelo ficou *confiantemente errado*.
   Com 3e-5, `mask_rate` final 0,625.
4. **O ganho não vem da aumentação.** Rodamos um controle (`p0_finetune_weak`): o
   piso com a mesma aumentação fraca do FixMatch. Ele faz 0,482, **acima** do piso
   original e de todas as variantes de FixMatch. Sem esse controle teríamos atribuído
   ao método um efeito que é da aumentação.

**O que mudamos.** Dois pontos em que a IA estava indo pelo caminho fácil e
corrigimos:

- Ela ia reportar o viés de classe como o enunciado prevê — "os pseudo-rótulos vão
  para as classes grandes". Exigimos o número. A correlação de Pearson entre suporte
  real e razão de viés é **+0,125**: praticamente nula. A maior classe
  (`american_black_bear`, 82 imagens) recebe **três vezes menos** pseudo-rótulos do
  que mereceria, e `eastern_chipmunk` recebe 4,2 vezes mais. O viés é **por
  confiança, não por frequência**, e é assim que vamos apresentar — contrariando a
  previsão do enunciado, com o número na tela.
- A função `bias_ratio` existia só nos testes, sem ninguém chamar no pipeline: os
  números do diagnóstico 2 só existiam dentro de uma figura. Ligamos ela ao
  relatório (`outputs/p1/vies_classes.csv`), porque número que só existe em PNG não
  dá para citar nem conferir.

**Defeito que a Parte 2 teria exposto calado.** Ao preparar a curva de orçamento
descobrimos que os relatórios das Partes 0 e 1 procuravam execuções por
`<nome>_b*_s*`, sem filtrar o orçamento. Como a Parte 2 cria `..._b1_...` e
`..._b20_...` no mesmo `outputs/`, a tabela que afirma reportar o orçamento de 5
passaria a tirar média sobre três orçamentos diferentes — **sem erro nenhum**. O
filtro foi unificado em `src/tracking.py:find_runs`, com teste que amarra as duas
pontas da convenção (`run_name` grava, `find_runs` procura).

---

## Episódio 4 — Parte 3: a IA ia me deixar classificar as fotos sem perceber (14/11)

**Problema.** O enunciado pede, para cada erro do modelo final, "um diagnóstico
concreto", e dá o exemplo: *"foto noturna, o animal ocupa 40×30 pixels depois do
redimensionamento para 224, e o mapa está todo no tronco da árvore ao fundo"*.

**Como a IA foi usada.** Pedi a implementação da Parte 3 inteira. O primeiro desenho
que discutimos produzia exatamente a frase do enunciado — e foi aí que o problema
apareceu: *"o mapa está no tronco da árvore ao fundo"* só pode ser escrito por quem
**olhou a foto e disse o que havia nela**. Para a IA, isso é classificar imagem do
conjunto, que o `CLAUDE.md` proíbe; e o "animal ocupa 40×30 pixels" exige a caixa do
animal, que o pacote não traz.

**O que validamos, e o que mudou.** O diagnóstico passou a ser montado só de
comparações mensuráveis, cada etiqueta com o número ao lado:

- **noite** — saturação média da imagem abaixo de 25 (câmera trap noturna usa
  infravermelho e grava quase sem cor);
- **atenção difusa** — fração da massa do mapa nos 10% de pixels mais quentes,
  comparada com a **mediana das 811 imagens do `test_public`**, não com um número
  escolhido a mão;
- **mapa na borda** — massa na faixa externa contra a área que essa faixa ocupa;
- **olhou para outro lugar** — IoU entre a região do mapa da classe **prevista** e a
  da classe **verdadeira**. Essa coluna separa dois modos de falha que a matriz de
  confusão junta: IoU alto é olhar certo e nomear errado; IoU baixo é olhar para
  outro lugar;
- **"animal pequeno" não existe como etiqueta.** Em lugar dela, "região usada
  pequena", que fala do mapa e não do animal.

A leitura visual ficou explicitamente para mim, na apresentação, com a figura e o
gabarito do pacote na mão. O arquivo `outputs/p3/galeria.md` diz isso em texto.

**Três coisas que a IA propôs e que melhoraram o trabalho.**

1. **CAM em vez de Grad-CAM, com prova em vez de alegação.** CAM está na aula 13
   (p. 80–82) e Grad-CAM não está em aula nenhuma. A IA mostrou que numa rede que
   termina em GAP + linear os dois dão o **mesmo mapa** a menos de constante, porque
   o peso do Grad-CAM se reduz a `w_k/Z`. Em vez de acreditar, pedimos teste: o
   `tests/test_saliency.py` calcula o Grad-CAM pelo autograd e compara os dois mapas.
2. **A oclusão virou teste do mapa, não um segundo mapa.** Um mapa de saliência é uma
   afirmação sobre a decisão; apagamos a região apontada e medimos a probabilidade.
   O que faz isso valer é o **controle**: apagar também uma região fria de área igual.
   Sem ele, qualquer queda se confundiria com "apagar 10% da imagem atrapalha". Nas
   oito imagens da galeria o relatório diz, com os dois números, se o mapa explica.
3. **A escolha das imagens virou regra determinística.** Escolher a dedo "os erros
   mais interessantes" depois de ver os mapas seria selecionar a evidência. Os três
   erros são um por par de confusão mais frequente, e dentro do par o mais confiante;
   os cinco acertos são os mais confiantes de classes distintas.

**Erro meu que a IA detectou sozinha, e que já tinha nos mordido.** A primeira versão
da comparação "acertos contra erros" usava a média bruta. Os erros se concentram nas
classes difíceis, então essa média pode estar medindo "classes diferentes têm mapas
diferentes". É o **mesmo confundidor** que na Parte 1 fez a hipótese da noite inverter
de sinal (`docs/decisions.md`, D19): bruto +12,8%, controlando a classe −6,5%. A
comparação da Parte 3 saiu com as duas colunas desde o começo — e o controle de classe
foi extraído para `src/analysis/groups.py`, para existir uma implementação só.

**Detalhe pequeno que teria produzido figura bonita e conclusão errada.** O controle
frio da oclusão é "os 10% de pixels mais frios", e num mapa normalizado isso empata em
milhares de pixels (tudo que é zero). Resolvendo o empate na ordem do array, o
controle cairia sempre na **faixa de cima da foto** — que em câmera trap é céu ou
folhagem, fácil de apagar sem custo nenhum. O teste de fidelidade sairia
artificialmente generoso, e todo mapa pareceria fiel. O empate passou a ser resolvido
pela distância ao pico, e há um teste que falha se o controle encostar na região
quente.
