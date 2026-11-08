# Status

Atualizado em 08/11/2025.

## Feito

- **Estrutura do pacote** (`src/`): configuração, dados, modelos, métodos,
  treino, avaliação, métricas e rastreamento. Decisões em `docs/decisions.md`.
- **Leitura do pacote de dados**: `classes.csv`, os três orçamentos e as três
  pastas de imagem, com os números conferidos por teste (6.505 / 202 / 811, e
  21 / 105 / 420 rótulos aninhados).
- **Barreira do `test_public`**: toda função de seleção recusa conjunto que não
  seja o `val`, com teste de contrato que cobre funções novas por introspecção.
- **Laço de treino** independente da estratégia, com checkpoint `best.pt`/`last.pt`,
  retomada por `--resume`, histórico em JSON e manifesto por execução
  (configuração, seed, commit).
- **Interface de entrega**: `predict.py --imagens PASTA --saida preds.csv`,
  CPU-only, pesos locais, imprime a contagem de parâmetros.
- **106 testes passando** (`make test`); 1 pulado de propósito (contrato das
  estratégias, que ativa quando a primeira estratégia for registrada).

## Em andamento

Nada. A estrutura está fechada e o próximo passo depende de uma decisão.

## Próximo passo

1. **Escolher a trilha (A, B ou C).** É o que destrava as Partes 1 a 3.
2. **Decidir o backbone da Parte 0 (b)** e registrá-lo em
   `src/models/registry.py`, com a estratégia de aumentação correspondente.
3. **Rodar a Parte 0** (piso): CNN pequena do zero e backbone ImageNet com
   fine-tuning, nos 105 rótulos; F1 macro e acurácia no `test_public`, matriz de
   confusão e curva de treino/validação da versão (b).

## Riscos

| Risco | Impacto | Mitigação |
|---|---|---|
| Trilha ainda não escolhida | M2 (13/11) depende dela | decidir antes de qualquer implementação de método |
| `val` com 202 imagens e 2 de cavalo | seleção de modelo ruidosa | tratar diferença pequena como ruído e sempre mostrar número por classe |
| Orçamento de 20 rotula quase metade do cavalo (45 no train) | F1 de cavalo instável na Parte 2 | reportar por classe, com o aviso do enunciado |
| 3 seeds × 3 orçamentos × 2 métodos = 18 treinos | prazo | pipeline completo antes de sofisticar; `--set` já permite varrer sem novo YAML |

## Bloqueios

- Nenhum bloqueio técnico. O pacote de dados está em `data/` e completo.

## Marcos

| Marco | Entrega | Meta | Situação |
|---|---|---|---|
| M0 | Estrutura, dados, testes e `predict.py` | 09/11 | estrutura, dados e testes feitos; `predict.py` precisa de um backbone registrado |
| M1 | Parte 0: piso, curvas e matriz de confusão | 10/11 | não iniciado |
| M2 | Parte 1: trilha escolhida e experimentos obrigatórios | 13/11 | não iniciado |
| M3 | Parte 2: curva de orçamento e ganho por classe | 14/11 | não iniciado |
| M4 | Parte 3: falhas, acertos e saliência | 15/11 | não iniciado |
| M5 | Ambiente limpo, README, pesos e `AI_LOG.md` | 16/11 12h | não iniciado |
