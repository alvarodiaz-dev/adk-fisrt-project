from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioConnectionParams
from mcp.client.stdio import StdioServerParameters 
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from getpass import getpass
from typing import Dict, List, Optional
import os
import json
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

            default_final_response = "El agente no produjo ninguna respuesta"

            async for event in runner.run_async(user_id = user_id, session_id= session_id, new_message= content):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        default_final_response = event.content.parts[0].text
                    elif event.actions and event.actions.escalate:
                        default_final_response = f"El agente escalo: {event.error_message or 'Sin mensaje'}"
                    break
            print(f"AI response: {default_final_response}")

    WORK_DIR = "./agents/adk-mcp/test_folder"
    os.makedirs(WORK_DIR, exist_ok=True)

    files_to_create = {
        "readme.txt": "Hola, estoy conectando MCP con mi agente ADK",
        "data.json": json.dumps({"nombre": "ADK", "version": "1.0", "features": ["agents", "tools", "mcp"]}, indent=2),
        "lista_compras.txt": "- Leche\n- Pan\n- Huevos\n- Café\n- Frutas",
        "notas.md": "# Notas del Curso\n\n## MCP\n- Model Context Protocol\n- Integración con ADK\n- Ejemplos prácticos"
    }

    for filename, content in files_to_create.items():
        with open(os.path.join(WORK_DIR, filename), "w") as f:
            f.write(content)

    print(f"Directorio de trabajo creado: {WORK_DIR}")
    print("\nArchivos creados:")
    for filename in os.listdir(WORK_DIR):
        print(f" - {filename}")

    filesystem_agent = LlmAgent(
        model = "gemini-2.5-flash",
        name = "filesystem_assistant",
        description = "Asistente para gestión de archivos usando MCP",
        instruction=f"""
            Eres un asistente experto en gestión de archivos.
            Puedes listar archivos, leer su contenido y ayudar al usuario
            a organizar su información. Trabajas con el directorio:
            {WORK_DIR}
            """,
        tools=[
            McpToolset(
                connection_params = StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command="npx",
                        args=[
                            "-y",
                            "@modelcontextprotocol/server-filesystem",
                            os.path.abspath(WORK_DIR)
                        ]
                    ), 
                    timeout=60
                )
            )
        ],
        generate_content_config = types.GenerateContentConfig(
            temperature= 0.1,
            max_output_tokens=800
        )
    )

    session_service = InMemorySessionService()

    APP_NAME = "mcp_filesystem_tutorial"
    USER_ID = "user_1"
    SESSION_ID = "session_001"

    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    runner = Runner(
        app_name=APP_NAME,
        session_service= session_service,
        agent=filesystem_agent
    )

    await call_agent_async(
        query="Que dice el readme.txt?",
        runner=runner,
        user_id=USER_ID,
        session_id= SESSION_ID
    )

if __name__ == "__main__":
    asyncio.run(main())

