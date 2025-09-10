from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from schema import ResponseModel


# --- Configuração API KEY ---
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


# --- Configurar o PROMPT do modelo (prompt de sistema) ---
prompt = (
"Você é um triador de Service Desk para políticas internas da empresa Carraro Desenvolvimento. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
    "{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
    '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"
    "Regras:\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex: "Posso reembolsar a internet do meu home office?", "Como funciona a política de alimentação em viagens?").\n'
    '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou contexto (Ex: "Preciso de ajuda com uma política", "Tenho uma dúvida geral").\n'
    '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial, ou quando o usuário explicitamente pede para abrir um chamado (Ex: "Quero exceção para trabalhar 5 dias remoto.", "Solicito liberação para anexos externos.", "Por favor, abra um chamado para o RH.").'
    "Analise a mensagem e decida a ação mais apropriada."
)


# --- Configuração Modelo Gemini ---
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0,
    api_key=GEMINI_API_KEY
)


# --- Configuração do modelo de saída da função triagem (a qual abriremos os chamados) ---

from langchain_core.messages import SystemMessage, HumanMessage

llm_saida_estruturada = llm.with_structured_output(ResponseModel)


# --- Definindo função de triagem ---

def triagem(mensagem: str) -> dict:
    saida: ResponseModel = llm_saida_estruturada.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=mensagem)
    ])
    
    return saida.model_dump() 

    # transforma a saida (ResponseModel - objeto JSON) em um dict python

print(triagem("Tenho quantos dias de férias?"))

print(triagem("Quero 5 dias de folga na semana que vem."))

print(triagem("Quem é o diretor da empresa?"))

print(triagem("Quem descobriu o Brasil?"))