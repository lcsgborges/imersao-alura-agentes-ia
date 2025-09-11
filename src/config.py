from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
GoogleGenerativeAIEmbeddings, 
ChatGoogleGenerativeAI
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pathlib import Path


# --- Configuração API KEY ---
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


# --- Configuração Modelo Gemini ---
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0,
    api_key=GEMINI_API_KEY
)


# --- Ler os pdf e salvar numa lista docs ---
DOC_PATH = Path(__file__).parent.parent / 'politicas/'

docs = []

for pdf in os.listdir(DOC_PATH):
    pdf = DOC_PATH / pdf
    try:
        loader = PyMuPDFLoader(pdf)
        docs.extend(loader.load())
        print(f'Success in file {pdf}')
    except Exception as e:
        print(f'Error in {pdf}: {e}')


# --- Configurar os chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

chunks = splitter.split_documents(docs)


# --- Configurar o modelo de embedding ---
embeddings = GoogleGenerativeAIEmbeddings(
    model='models/gemini-embedding-001',
    google_api_key=GEMINI_API_KEY
)


# --- Transformar nossos chunks em vetores utilizando o modelo de embedding ---
vectorstore = FAISS.from_documents(chunks, embeddings)


# --- Fazer a busca/comparação entre os vetores
retriever = vectorstore.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'score_threshold': 0.5, 'k':4
})


# --- Criando o prompt de sistema para nosso modelo ---

prompt_rag = ChatPromptTemplate([
    ("system",
     "Você é um Assistente de Políticas Internas (RH/IT) da empresa Carraro \
     Desenvolvimento. "
     "Responda SOMENTE com base no contexto fornecido. "
     "Se não houver base suficiente, responda apenas 'Não sei'."),

    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])


# --- Esta chain pega os documentos relevantes retornados pelo retriever, 
# insere-os no template de prompt junto com a pergunta do usuário, e envia tudo 
# para o modelo de linguagem ---

document_chain = create_stuff_documents_chain(llm, prompt_rag)


# --- Criar função de perguntar pro modelo ---

def question_model(question: str) -> dict:
    docs_of_context = retriever.invoke(question)
    
    if not docs_of_context:
        return {
            'answer': 'Não sei.',
            'cites': [],
            'context': False
        }
    
    answer = document_chain.invoke({'input': question, 'context': docs_of_context})
    
    txt = (answer or '').strip()
    
    if txt.rstrip('.!?') == 'Não sei':
        return {
            'answer': 'Não sei.',
            'cites': [],
            'context': False
        }
    
    return {
        'answer': txt,
            'cites': docs_of_context,
            'context': True
    }