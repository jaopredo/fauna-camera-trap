"""Diagnosticos e figuras das Partes 1 a 3 — o que nao pertence a trilha nenhuma.

Separado de ``src.metrics`` de proposito: ``metrics`` devolve numero e nao importa
matplotlib; aqui ficam as figuras, as tabelas e as analises.

    figures.py    curvas de treino, matriz de confusao, curva de orcamento, galeria
    budget.py     a curva de orcamento da Parte 2 e o teto de referencia
    saliency.py   CAM, medidas do mapa e a oclusao como teste de fidelidade
    gallery.py    quais imagens entram na galeria da Parte 3 e o diagnostico medido
    groups.py     comparar uma medida entre dois grupos controlando a classe
    images.py     medidas tiradas da imagem crua (proxy de foto noturna)

Nada aqui importa de ``src.tracks``: os tres ultimos modulos existem justamente
porque a Parte 3 precisa de medidas que nasceram numa trilha, e apagar a trilha nao
pode leva-las embora (``docs/decisions.md``, D18 e D21). Os modulos sao importados
pelo caminho completo, sem reexportacao, para que fique claro de onde cada coisa vem.
"""
