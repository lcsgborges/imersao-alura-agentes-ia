# Configurar o modelo/formato/tipagem da resposta

from pydantic import BaseModel, Field
from typing import Literal, List


class ResponseSchema(BaseModel):
    decisao: Literal['AUTO_RESOLVER', 'PEDIR_INFO', 'ABRIR_CHAMADO']
    urgencia: Literal['BAIXA', 'MEDIA', 'ALTA']
    campos_faltantes: List[str] = Field(default_factory=list)
    
    # O uso de Field(default_factory=list) garante que cada instância do modelo
    # tenha sua própria lista, evitando problemas de mutabilidade.