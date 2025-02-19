import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


dados = {
    "inscrição": [
        "Você pode se inscrever através do nosso site ou visitar um de nossos centros locais. O processo de inscrição leva cerca de 15 minutos.",
        "Qual é o processo de inscrição no curso?",
        "Onde posso me inscrever para o curso?",
        "Como faço para me inscrever no curso?",
        "Quais são os passos para me matricular?"
    ],
    "horário": [
        "As aulas estão disponíveis nos turnos da manhã (9-11h), tarde (14-16h) e noite (19-21h). Você pode escolher o horário que melhor se encaixa na sua agenda.",
        "Quais são os horários das aulas?",
        "Quando as aulas acontecem?",
        "Quais os horários disponíveis para as aulas?",
        "Os horários das aulas são flexíveis?"
    ],
    "formato": [
        "Sim, oferecemos aulas tanto online quanto presenciais. As aulas online utilizam nossa plataforma virtual com interação ao vivo com instrutores.",
        "O curso é online ou presencial?",
        "Como são as aulas? Online ou presenciais?",
        "Quais são os formatos das aulas?",
        "As aulas são presenciais ou à distância?"
    ],
    "materiais": [
        "O curso inclui livros digitais, cadernos de exercícios, materiais de áudio e acesso à nossa plataforma de aprendizagem online.",
        "Quais materiais estão incluídos no curso?",
        "O que é fornecido como material do curso?",
        "O curso oferece algum material didático?",
        "Há materiais extras fornecidos no curso?"
    ]
}


def initialize_model():
    # Carregar modelo de embeddings em português
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def create_index(model, dados):
    # Gerar embeddings para as perguntas e respostas
    all_texts = []  # Para armazenar todas as perguntas
    all_responses = []  # Para armazenar as respostas
    for key, value in dados.items():
        for item in value:
            all_texts.append(item)
            all_responses.append(dados[key][0])  # Pega a primeira resposta associada
    
    embeddings = model.encode(all_texts, convert_to_tensor=False)
    embeddings = np.array(embeddings).astype('float32')
    
    # Criar índice FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index, all_texts, all_responses


def get_most_similar(model, index, query, all_texts, all_responses, threshold=0.7):
    # Gerar embedding para a pergunta
    query_vector = model.encode([query], convert_to_tensor=False)
    query_vector = np.array(query_vector).astype('float32')
    
    # Buscar texto mais similar
    distances, indices = index.search(query_vector, k=1)
    
    if distances[0][0] < threshold:
        return all_responses[indices[0][0]]  # Retorna a resposta associada à pergunta
    else:
        return "Desculpe, não encontrei uma resposta adequada para sua pergunta."


def main():
    try:
        print("Inicializando o modelo... (isso pode levar alguns segundos)")
        model = initialize_model()
        
        # Preparar textos e criar índice
        index, all_texts, all_responses = create_index(model, dados)
        
        print("Bem-vindo ao nosso chatbot!")
        nome = input("Qual seu nome? ")
        print(f"Muito prazer, {nome}! Eu vou te ajudar com suas perguntas sobre o curso!")

        while True:
            entrada_usuario = input("Você: ")
            if entrada_usuario.lower() == "sair":
                break

            try:
                resposta = get_most_similar(model, index, entrada_usuario, all_texts, all_responses)
                print("Bot:", resposta)
            except Exception as e:
                print(f"Erro ao processar pergunta: {e}")
                print("Bot: Desculpe, ocorreu um erro ao processar sua pergunta.")

    except Exception as e:
        print(f"Erro ao inicializar o chatbot: {e}")


if __name__ == "__main__":
    main()


#aplicar o uso do tensor flow e engenharia de prompt tal qual um DB maior pra treinar interpretação e nuances