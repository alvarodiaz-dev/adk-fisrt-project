import asyncio
import os
from google.adk.agents import LoopAgent, LlmAgent, BaseAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.tools.tool_context import ToolContext

GEMINI_MODEL = "gemini-2.0-flash"

def exit_loop(tool_context: ToolContext):
  """Call this function ONLY when the critique indicates no further changes are needed, signaling the iterative process should end."""
  print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
  tool_context.actions.escalate = True
  return {}

print("\n=== Ejemplo: LoopAgent ===")

escritor = LlmAgent(
    name="Escritor",
    model="gemini-2.0-flash",
    instruction=(
        "Mejora el texto actual basándote en las críticas recibidas. "
        "Si no hay texto previo, genera uno inicial sobre el tema dado."
    ),
    output_key="texto_actual"
)

critico = LlmAgent(
    name="Critico",
    model="gemini-2.5-flash",
    instruction=(
        "Revisa de manera muy detallada y critica el siguiente texto 'texto_actual'"
        "y proporciona críticas constructivas para mejorarlo. "
        "Dado el caso que el texto sea excelente y no necesite de ninguna mejora, di exactamente: "
        "'No se requieren más mejoras.' y llama a la herramienta exit_loop. "
        "De lo contrario, proporciona sugerencias específicas para mejorar el texto."
    ),
    tools=[exit_loop],
    output_key="critica"
)


loop_refinamiento = LoopAgent(
    name="LoopRefinamiento",
    sub_agents=[escritor, critico],
    max_iterations=3 
)

root_agent = loop_refinamiento