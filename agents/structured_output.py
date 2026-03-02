from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio

load_dotenv()

async def main():
    async def call_agent_async(query: str, runner, user_id, session_id):
        """Envía una consulta al agente e imprime la respuesta final"""

        print(f"Request del usuario: {query}")

        content = types.Content(
            role="user",
            parts=[
                types.Part(text=query)
            ]
        )

        final_response = "El agente no produjo ninguna respuesta"

        async for event in runner.run_async(user_id = user_id, session_id= session_id, new_message= content):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response = f"El agente escalo: {event.error_message or 'Sin mensaje'}"
                break
        print(f"AI response: {final_response}")
    
    MODEL_GEMINI = "gemini-2.5-flash"
    MODEL_OPENAI = LiteLlm("openai/gpt-5-nano")
    MODEL_ANTHROPIC = LiteLlm("anthropic/claude-3-haiku-20240307")

    class InformacionProducto(BaseModel):
        """Esquema para extraer informacion de productos"""
        nombre: str = Field(description="Nombre completo del producto")
        marca: Optional[str] = Field(description="Marca del producto")
        precio: Optional[float] = Field(description="Precio del producto en soles")
        caracteristicas: List[str] = Field(
            default_factory= list,
            description="Lista de caracteristicas principales del producto"
        )
        disponible: bool = Field(description="Si esta disponible")
        categoria: Optional[str] = Field(description="Categoria del producto")

    class SentimentAnalysis(BaseModel):
        """Esquema para analisis sentimental en texto"""
        sentimiento: str = Field(
            description="Sentimiento general: positivo, negativo or neutral"
        )
        confianza: float = Field(
            description="Nivel de confianza del analisis (0.0 a 1.0)",
            ge=0.0, le=1.0
        )
        emociones: List[str] = Field(
            default_factory=list,
            description="Emocionaes detectadas en el texto"
        )
        aspectos_positivos: List[str] = Field(
            default_factory=list,
            description="Aspectos positivos mencionados"
        )
        aspectos_negativos: List[str] = Field(
            default_factory=list,
            description="Aspectos negativos mencionados"
        )
    
    class Event(BaseModel):
        """Informacion acerca de un evento"""
        titulo: str = Field(description="Titulo del evento")
        fecha: Optional[str] = Field(None, description="Fecha del evento (YYYY-MM-DD)")
        hora: Optional[str] = Field(None, description="Hora del evento (HH:MM)")
        ubicacion: Optional[str] = Field(None, description="Ubicación del evento")
        participantes: List[str] = Field(
            default_factory=list,
            description="Lista de participantes"
        )
        descripcion: Optional[str] = Field(None, description="Descripcion del evento")

    class EventList(BaseModel):
        """Lista de eventos obtenidos"""
        eventos: List[Event] = Field(
            default_factory=list,
            description="Lista de todos los eventos encontrados"
        )
        eventos_totales: int = Field(
            description="Numero total de eventos encontrados"
        )

    products_agent = LlmAgent(
        model=MODEL_GEMINI,
        name="ExtractorProductos",
        description="Extrae información extructura de productos",
        output_schema= InformacionProducto,
        output_key = "InformacionProducto",
        generate_content_config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1000
        ),
        instruction=("""Extrae información de productos del texto proporcionado.
        Sigue exactamente el esquema definido.
        Si no encuentras ningún dato, usa un valor nulo o una lista vacía según corresponda.
        Sé preciso con los precios y características.
        De haber fallos ortograficos, corrigelos y agregalos corregidos
        """)
    )

    session_service = InMemorySessionService()

    APP_NAME = "test_gemini"
    USER_ID = "user_1"
    SESSION_ID = "thread_001"

    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id= USER_ID,
        session_id= SESSION_ID
    )

    runner = Runner(
        agent=products_agent,
        app_name= APP_NAME,
        session_service= session_service
    )

    texto = """
    El nuevo IPhone 15 PRO MAX de apple ya esta disponible.
    Con un precio de 2.300 soles, incluye camara de 48 MP, pantalla de 6.7 pulgadas,
    chip A17 Pro y bateria de larga duración. Disponible en titanio.
    """

    await call_agent_async(
        query=texto,
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

if __name__ == "__main__":
    asyncio.run(main())