from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from pathlib import Path
from schema import ResponseSchema


# --- Configuração API KEY ---
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


# --- Configurar o PROMPT do modelo (prompt de sistema) ---
PROMPT_PATH = Path(__file__).parent.parent / 'prompts.txt'

with open(PROMPT_PATH, 'r') as prompt:
    system_prompt = prompt.read()


# --- Configuração Modelo Gemini ---
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0,
    api_key=GEMINI_API_KEY
)


# --- Configuração do modelo de saída da resposta---
llm_structured_output = llm.with_structured_output(ResponseSchema)


# --- Definindo função de pergunta para o modelo criado ---
def send_human_prompt(mensagem: str) -> dict:
    saida: ResponseSchema = llm_structured_output.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=mensagem)
    ])
    
    return saida.model_dump() 

# model_dump(): transforma a saida (ResponseSchema) em um dict python