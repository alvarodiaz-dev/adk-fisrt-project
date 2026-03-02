from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
import asyncio
from dotenv import load_dotenv

load_dotenv()

agent_search = Agent(
    name = "AgenteBuscador",
    model= "gemini-2.5-flash",
    description= "Un agente amigable que puede buscar información en de google en tiempo real",
    tools= [google_search],
    instruction=(
        "Eres un asistente amigable y divertido"
        "Buscas información actualizada de paginas confiables de google"
        "Brindas informacion detallada segun la información que hayas encontrado en google"
        "Cuando se te haga una pregunta buscaras en google si asi es requerido"
        "Proporcionas respuestas directas, claras y concisas"
        "Si no estas seguro de tu respuesta, busca en google en busca de información actualizada"
        "Siempre responde de manera amigable y directa"
    )
)

async def main():
    session_service = InMemorySessionService()

    APP_NAME = "primera_app_con_adk"
    USER_ID = "user_1"
    SESSION_ID = "thread_001"

    session = await session_service.create_session(
        app_name= APP_NAME,
        user_id= USER_ID,
        session_id= SESSION_ID
    )

    print(f"Sesión creada: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

    runner = Runner(
        agent=agent_search,
        app_name=APP_NAME,
        session_service= session_service
    )

    print(f"Runner creado para el agente: '{runner.agent.name}'")

    events = runner.run(
        user_id=USER_ID,
        session_id= SESSION_ID,
        new_message= types.Content(
            role = "user", 
            parts=[
                types.Part(text="¿Cuales son las ultimas noticias sobre framworks de IA?")
            ]
        )
    )

    # for event in events:
    #     if event.is_final_response():
    #         if event.grounding_metadata.grounding_chunks:
    #             for _ in event.grounding_metadata.grounding_chunks:
    #                 print(f"Grounding Chunk: '{_.web.title}")
    #         else:
    #             print("No es necesario el uso de grounding")
    #         final_response = event.content.parts[0].text
    #         print(f"Respuesta: '{final_response}'")

    async def call_agent_async(query: str, runner, user_id, session_id):
        """Envía una consulta al agente e imprime la respuesta final"""

        print(f"Request del usuario: {query}")

        content = types.Content(
            role="user",
            parts=[
                types.Part(text=query)
            ]
        )

        default_final_response = "El agente no produjo ninguna respuesta"

        async for event in runner.run_async(user_id = user_id, session_id= session_id, new_message= content):
            if event.is_final_response():
                if event.content and event.content.parts:
                    default_final_response = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    default_final_response = f"El agente escalo: {event.error_message or 'Sin mensaje'}"
                break
        print(f"AI response: {default_final_response}")

    await call_agent_async(
        query="¿Cuales son las ultimas noticias sobre framworks de IA?",
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

if __name__ == "__main__":
    asyncio.run(main())
    