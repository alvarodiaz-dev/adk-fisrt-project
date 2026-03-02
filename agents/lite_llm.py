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

    refranes_agent = LlmAgent(
        model=MODEL_GEMINI,
        name="refranes_agent",
        description="Completa los refranes que el usuario empieza",
        generate_content_config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=1000,
            top_k= 40
        )
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
        agent=refranes_agent,
        app_name= APP_NAME,
        session_service= session_service
    )

    await call_agent_async(
        query="Camaron que se duerme...",
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

if __name__ == "__main__":
    asyncio.run(main())