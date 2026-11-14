# Status

Atualizado em 12/11/2025.

## Feito

- **Estrutura do pacote** (`src/`): configuração, dados, modelos, estratégias,
  treino, avaliação, métricas, análise e rastreamento. Decisões em
  `docs/decisions.md`; padrões documentados no `README.md`.
- **Barreira do `test_public`**: toda função de seleção recusa conjunto que não seja
  o `val`, com teste de contrato que cobre funções novas por introspecção.
- **Parte 0 completa**, 3 seeds por versão, orçamento de 5 rótulos.
- **151 testes passando** (`make test`), `ruff` limpo.
- `docs/aulas_index.md` — índice das aulas, com o que **não** está nelas.

### Números da Parte 0 (`test_public`, média ± desvio sobre 3 seeds)

| Versão | Parâmetros | F1 macro | Acurácia |
|---|---:|---|---|
| (a) `small_cnn` do zero | 394.293 | **0,251 ± 0,008** | 0,261 ± 0,022 |
| (b) ResNet-18 ImageNet + fine-tuning | 11.187.285 | **0,464 ± 0,036** | 0,485 ± 0,024 |
| teto de referência (monitor, todos os rótulos) | — | 0,686 | — |

O ImageNet recupera **49% da distância** entre o piso do zero e o teto. A Parte 1
disputa os 0,222 de F1 que sobram, sem pedir mais rótulos.

Artefatos: `outputs/p0/tabela.md`, `f1_por_classe.csv`, `confusao_finetune.png`,
`curva_finetune.png`; `curva_orcamento.csv` já com as 6 linhas do orçamento de 5.

### Inferência

`predict.py` classifica as 811 imagens do `test_public` em **15,6 s** em CPU. O
limite do enunciado é 10 min para ~800 imagens — margem de ~38×.

## Em andamento

Nada. A Parte 0 está fechada e o próximo passo depende de uma decisão.

## Próximo passo

1. **Escolher a trilha (A, B ou C).** É o que destrava as Partes 1 a 3.
2. Parte 2: faltam os orçamentos de 1 e 20 para o piso (uma seed cada).
3. Parte 3: galeria de falhas e saliência.

## Riscos

| Risco | Impacto | Mitigação |
|---|---|---|
| Trilha não escolhida; M2 venceu 13/11 | bloqueia Partes 1 a 3 | decidir antes de qualquer implementação de método |
| Época escolhida no `val` é instável | a "melhor época" é ruído | 13 épocas ficam a menos de 0,01 do melhor (seed 0); melhores épocas por seed: 54, 38, 18. Tratar como platô, não como escolha |
| `horse` tem 4 imagens no `test_public` | F1 de 0,785 oscila com um acerto a mais ou a menos | sempre reportar por classe, com o suporte |
| Treino em CPU (sem GPU na máquina) | Trilha A (SSL em 6.505 imagens) é inviável localmente | usar Colab para a Parte 1 se a trilha for A |

## Bloqueios

- Nenhum bloqueio técnico.

## Marcos

| Marco | Entrega | Meta | Situação |
|---|---|---|---|
| M0 | Índice das aulas, estrutura, dados, testes e `predict.py` | 09/11 | **feito** (índice entregue em 12/11) |
| M1 | Parte 0: piso, curvas e matriz de confusão | 10/11 | **feito** em 12/11, com atraso |
| M2 | Parte 1: trilha escolhida e experimentos obrigatórios | 13/11 | **não iniciado** — trilha em branco |
| M3 | Parte 2: curva de orçamento e ganho por classe | 14/11 | parcial: o ponto de 5 rótulos do piso já existe |
| M4 | Parte 3: falhas, acertos e saliência | 15/11 | não iniciado |
| M5 | Ambiente limpo, README, pesos e `AI_LOG.md` | 16/11 12h | não iniciado |

**Se o prazo apertar**, cortar nesta ordem: o orçamento de 1 rótulo (a Parte 2 fica
com 5 e 20), depois os 5 acertos com saliência da Parte 3 (ficam os 3 erros, que são
o obrigatório). Nunca cortar: os experimentos obrigatórios da trilha, a curva de
orçamento e o `predict.py`.
