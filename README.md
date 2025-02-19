---

📚 Chatbot Inteligente com Busca Vetorial

Este projeto implementa um chatbot de perguntas e respostas baseado em busca vetorial e algoritmos de similaridade semântica. Ele utiliza FAISS para indexação eficiente e Sentence Transformers para geração de embeddings, permitindo respostas inteligentes baseadas no significado das perguntas.

🚀 Tecnologias Utilizadas

Python

FAISS (Facebook AI Similarity Search)

Sentence Transformers

NumPy

TensorFlow (planejado para futuras melhorias)


📌 Como Funciona?

🔍 Busca Vetorial

A busca vetorial permite encontrar textos semelhantes comparando seus vetores de representação. O processo funciona assim:

1. Cada pergunta no banco de dados é convertida em um vetor de alta dimensão usando Sentence Transformers.


2. Esses vetores são armazenados em um índice otimizado com FAISS.


3. Quando o usuário faz uma pergunta, o chatbot gera um vetor para a entrada e busca no índice o vetor mais próximo.


4. A resposta associada ao vetor mais semelhante é retornada ao usuário.



📏 Algoritmo de Similaridade

Este projeto usa a métrica de distância euclidiana () para encontrar perguntas semelhantes. O FAISS permite buscar vetores próximos de forma eficiente, reduzindo o tempo de resposta.

Outras métricas comuns para busca vetorial incluem:

Cosseno da similaridade: Mede o ângulo entre os vetores (útil quando a escala não importa).

Distância de Manhattan (): Soma das diferenças absolutas entre as dimensões dos vetores.

Distância de Jaccard: Usada para comparar conjuntos de palavras em textos curtos.


Neste projeto, a distância euclidiana foi escolhida porque funciona bem com embeddings de Sentence Transformers, capturando diferenças semânticas de maneira eficiente.
