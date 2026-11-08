"""Base comum das excecoes do pacote.

Existe para os pontos de entrada: ``scripts/`` e ``predict.py`` capturam
``AppError`` e saem com uma linha de mensagem, em vez de despejar traceback para
um erro previsto (config errada, pacote de dados ausente, backbone inexistente).
Erro nao previsto continua subindo com traceback, que e o que se quer depurar.

Cada excecao tambem herda do tipo embutido que descreve a natureza do problema
(``ValueError`` para entrada invalida, ``RuntimeError`` para estado invalido), de
modo que quem captura pelo tipo padrao continua funcionando.
"""

from __future__ import annotations


class AppError(Exception):
    """Erro previsto do projeto, com mensagem apresentavel ao usuario."""
