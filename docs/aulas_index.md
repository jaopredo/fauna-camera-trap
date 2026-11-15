# Índice das aulas

Mapa de tópico → arquivo e páginas, para localizar o conteúdo antes de se basear
nele. **O índice não substitui a leitura**: as páginas abaixo foram abertas e lidas
antes de serem citadas em `docs/decisions.md`.

Arquivos em `docs/` (fora do Git por tamanho; ver `.gitignore`).

## O que está disponível

| Arquivo | Páginas | Assunto |
|---|---:|---|
| `DL_20242_11_Autoencoders.pdf` | 70 | autoencoders, convolucionais, denoising, variacionais |
| `DL_20242_12_Similarity.pdf` | 87 | one-shot learning, redes siamesas, similarity learning |
| `DL_20242_13_Unsupervised.pdf` | 95 | aprendizado com dados não rotulados: deep clustering, auto-supervisão, rótulos ruidosos |
| `DL_20242_14_Self_Attention.pdf` | 64 | self-attention, transformers, atenção para visão |
| `DL_20242_15_VisualFM.pdf` | 110 | foundation models visuais: ViT, Swin, MAE, MSN |
| `A2.pdf` | 53 | **resumo do próprio aluno**, não é aula — ver ressalva abaixo |

## Por tópico

### Fine-tuning de modelo pré-treinado
- `15_VisualFM`, **p. 15** — a receita em três passos: remover a cabeça de predição
  pré-treinada; acoplar uma camada nova, inicializada com zeros, com tantas saídas
  quantas as classes da tarefa; fazer o fine-tuning nos dados disponíveis. Enunciada
  para ViT (fonte: Dosovitskiy 2021).
- `15_VisualFM`, **p. 16** — fine-tuning em resolução maior que a do pré-treino:
  manter o tamanho do patch e interpolar em 2D os position embeddings.
- `15_VisualFM`, **p. 78** — fine-tuning contrastado com *visual prompting*, que
  adapta sem atualizar pesos.

### Arquitetura de backbone
- `13_Unsupervised`, **p. 60** — experimento do professor: "ResNet18 as backbone
  architecture followed by a projection head", treinado com SGD, entrada 32×32,
  aumentações do SwAV.
- `13_Unsupervised`, **p. 26** — encoder do SimCLR: ResNet-50 em três larguras; a
  representação é a saída depois do average pooling.
- `15_VisualFM`, **p. 2–6, 21, 24** — backbones transformer (ViT, Swin).

### Aumentação
- `13_Unsupervised`, **p. 25** — módulo de aumentação estocástica do SimCLR, em três
  passos: recorte aleatório com redimensionamento de volta ao tamanho original;
  distorção de cor; desfoque gaussiano.
- `13_Unsupervised`, **p. 17, 20, 22, 39** — aumentação no contexto de aprendizado
  contrastivo e semi-supervisionado.

### Aprendizado com dados não rotulados (Parte 1)
- `13_Unsupervised`, **p. 2–3** — o panorama que organiza a disciplina: deep
  clustering, auto-supervisão, rótulos ruidosos e transfer learning, entre dados não
  rotulados e supervisionados.
- `13_Unsupervised`, **p. 25–32** — SimCLR (Trilha A).
- `13_Unsupervised`, **p. 60–61** — SwAV (Trilha A).
- `13_Unsupervised`, **p. 85–91** — FixMatch (Trilha B): definição, razão mu e peso
  lambda, CE na vista fraca, pseudo-rótulo duro com limiar, soma ponderada.
- `15_VisualFM` — foundation models (Trilha C).

### Localização a partir de rótulo de classificação (Parte 3)
- `13_Unsupervised`, **p. 77–82** — *Weakly-Supervised Learning*: treinar com
  rótulos mais baratos do que a tarefa exige. **A p. 80–82 traz o CAM** (*Class
  Activation Mapping*), citando `Learning Deep Features for Discriminative
  Localization`, Zhou et al., CVPR 2016. A receita da aula: treinar a rede de
  classificação, fixar as camadas convolucionais, aprender o peso da contribuição de
  cada mapa de ativação para a classe, e somar as contribuições para obter o esboço
  das zonas ativas.
- **Consequência para a Parte 3:** o mapa de saliência sai de **CAM, não Grad-CAM**.
  CAM exige *global average pooling* seguido de camada linear, e a ResNet-18 tem
  exatamente isso (`avgpool` → `fc`): os pesos de `fc` **são** os `w_k^c` da fórmula,
  então não há camada nova a treinar nem hook a instalar. Grad-CAM ficaria fora do
  material da disciplina sem necessidade.

## Sobre o `A2.pdf`

É o **resumo escrito pelo próprio aluno** para a prova A2, não material do professor.
Serve para saber *o que ele já domina* (ver a memória do projeto), e **não** como
fonte para citar em `docs/decisions.md`: ele mesmo avisou que "não é SUPER rigoroso"
e "talvez contenha alguns erros".

Cobre: autoencoders (PCA, AE, convolucional, denoising, VAE); similaridade (one-shot,
siamesas, contrastive, WDMC, triplet); dados não rotulados (DEC, Deep K-Means,
SimCLR, LLP, SELFIE, **CAM**, **FixMatch**); self-attention (QKV, multihead,
Transformers, SASA).

**A seção 5, de Visual Foundation Models, está vazia** — só o título. É exatamente o
conteúdo que a Trilha C precisaria, e é o maior buraco a cobrir na apresentação.

Onde o resumo e o PDF da aula divergirem, **vale o PDF da aula**. O ponto que mais
importava (CAM) foi conferido e estava correto.

## O que **não** está nas aulas disponíveis

Registrado para que nenhuma decisão cite aula que não existe:

- **Transfer learning supervisionado a partir do ImageNet** como tópico próprio
  (quais camadas congelar, taxa de aprendizado diferenciada, quantas épocas). O
  termo "transfer learning" aparece na `13_Unsupervised` apenas como rótulo de
  figuras sobre SimCLR e SwAV — isto é, transferência *depois de pré-treino
  auto-supervisionado*, não o caso supervisionado da Parte 0 (b).
- **Projeto de CNN convolucional do zero** (profundidade, largura, pooling) —
  Parte 0 (a).
- **Aumentação para treino supervisionado** (a da `13_Unsupervised` p. 25 é a do
  SimCLR, para pares contrastivos).
- **Escolha de otimizador e taxa de aprendizado.**

Esses temas estão, presumivelmente, nas **aulas 1 a 10**, que não foram
disponibilizadas em `docs/`. As decisões que dependem deles estão marcadas em
`docs/decisions.md` como **"fora do material da disciplina"**.
