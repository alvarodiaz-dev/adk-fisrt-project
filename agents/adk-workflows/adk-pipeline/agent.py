from dotenv import load_dotenv
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.agents import LlmAgent, SequentialAgent                                                                                                     

load_dotenv()

texto_entrada = """
Introducing a new, unifying DNA sequence model that advances regulatory variant-effect prediction and promises to shed new light on genome function — now available via API.

The genome is our cellular instruction manual. It’s the complete set of DNA which guides nearly every part of a living organism, from appearance and function to growth and reproduction. Small variations in a genome’s DNA sequence can alter an organism’s response to its environment or its susceptibility to disease. But deciphering how the genome’s instructions are read at the molecular level — and what happens when a small DNA variation occurs — is still one of biology’s greatest mysteries.

Today, we introduce AlphaGenome, a new artificial intelligence (AI) tool that more comprehensively and accurately predicts how single variants or mutations in human DNA sequences impact a wide range of biological processes regulating genes. This was enabled, among other factors, by technical advances allowing the model to process long DNA sequences and output high-resolution predictions.

To advance scientific research, we’re making AlphaGenome available in preview via our AlphaGenome API for non-commercial research, and planning to release the model in the future.

We believe AlphaGenome can be a valuable resource for the scientific community, helping scientists better understand genome function, disease biology, and ultimately, drive new biological discoveries and the development of new treatments.

How AlphaGenome works
Our AlphaGenome model takes a long DNA sequence as input — up to 1 million letters, also known as base-pairs — and predicts thousands of molecular properties characterising its regulatory activity. It can also score the effects of genetic variants or mutations by comparing predictions of mutated sequences with unmutated ones.

Predicted properties include where genes start and where they end in different cell types and tissues, where they get spliced, the amount of RNA being produced, and also which DNA bases are accessible, close to one another, or bound by certain proteins. Training data was sourced from large public consortia including ENCODE, GTEx, 4D Nucleome and FANTOM5, which experimentally measured these properties covering important modalities of gene regulation across hundreds of human and mouse cell types and tissues.


Play video
Animation showing AlphaGenome taking one million DNA letters as input and predicting diverse molecular properties across different tissues and cell types.

The AlphaGenome architecture uses convolutional layers to initially detect short patterns in the genome sequence, transformers to communicate information across all positions in the sequence, and a final series of layers to turn the detected patterns into predictions for different modalities. During training, this computation is distributed across multiple interconnected Tensor Processing Units (TPUs) for a single sequence.

This model builds on our previous genomics model, Enformer and is complementary to AlphaMissense, which specializes in categorizing the effects of variants within protein-coding regions. These regions cover 2% of the genome. The remaining 98%, called non-coding regions, are crucial for orchestrating gene activity and contain many variants linked to diseases. AlphaGenome offers a new perspective for interpreting these expansive sequences and the variants within them.

AlphaGenome’s distinctive features
AlphaGenome offers several distinctive features compared to existing DNA sequence models:

Long sequence-context at high resolution
Our model analyzes up to 1 million DNA letters and makes predictions at the resolution of individual letters. Long sequence context is important for covering regions regulating genes from far away and base-resolution is important for capturing fine-grained biological details.

Previous models had to trade off sequence length and resolution, which limited the range of modalities they could jointly model and accurately predict. Our technical advances address this limitation without significantly increasing the training resources — training a single AlphaGenome model (without distillation) took four hours and required half of the compute budget used to train our original Enformer model.

Comprehensive multimodal prediction
By unlocking high resolution prediction for long input sequences, AlphaGenome can predict the most diverse range of modalities. In doing so, AlphaGenome provides scientists with more comprehensive information about the complex steps of gene regulation.

Efficient variant scoring
In addition to predicting a diverse range of molecular properties, AlphaGenome can efficiently score the impact of a genetic variant on all of these properties in a second. It does this by contrasting predictions of mutated sequences with unmutated ones, and efficiently summarising that contrast using different approaches for different modalities.

Novel splice-junction modeling
Many rare genetic diseases, such as spinal muscular atrophy and some forms of cystic fibrosis, can be caused by errors in RNA splicing — a process where parts of the RNA molecule are removed, or “spliced out”, and the remaining ends rejoined. For the first time, AlphaGenome can explicitly model the location and expression level of these junctions directly from sequence, offering deeper insights about the consequences of genetic variants on RNA splicing.

State-of-the-art performance across benchmarks
AlphaGenome achieves state-of-the-art performance across a wide range of genomic prediction benchmarks, such as predicting which parts of the DNA molecule will be in close proximity, whether a genetic variant will increase or decrease expression of a gene, or whether it will change the gene’s splicing pattern.


Bar graph showing AlphaGenome’s relative improvements on selected DNA sequence and variant effect tasks, compared against results for the current best methods in each category.

When producing predictions for single DNA sequences, AlphaGenome outperformed the best external models on 22 out of 24 evaluations. And when predicting the regulatory effect of a variant, it matched or exceeded the top-performing external models on 24 out of 26 evaluations.

This comparison included models specialized for individual tasks. AlphaGenome was the only model that could jointly predict all of the assessed modalities, highlighting its generality. Read more in our preprint.

The benefits of a unifying model
AlphaGenome’s generality allows scientists to simultaneously explore a variant's impact on a number of modalities with a single API call. This means that scientists can generate and test hypotheses more rapidly, without having to use multiple models to investigate different modalities.

Moreover AlphaGenome’s strong performance indicates it has learned a relatively general representation of DNA sequence in the context of gene regulation. This makes it a strong foundation for the wider community to build upon. Once the model is fully released, scientists will be able to adapt and fine-tune it on their own datasets to better tackle their unique research questions.

Finally, this approach provides a flexible and scalable architecture for the future. By extending the training data, AlphaGenome’s capabilities could be extended to yield better performance, cover more species, or include additional modalities to make the model even more comprehensive.

“
It’s a milestone for the field. For the first time, we have a single model that unifies long-range context, base-level precision and state-of-the-art performance across a whole spectrum of genomic tasks.

Dr. Caleb Lareau, Memorial Sloan Kettering Cancer Center

A powerful research tool
AlphaGenome's predictive capabilities could help several research avenues:

Disease understanding: By more accurately predicting genetic disruptions, AlphaGenome could help researchers pinpoint the potential causes of disease more precisely, and better interpret the functional impact of variants linked to certain traits, potentially uncovering new therapeutic targets. We think the model is especially suitable for studying rare variants with potentially large effects, such as those causing rare Mendelian disorders.
Synthetic biology: Its predictions could be used to guide the design of synthetic DNA with specific regulatory function — for example, only activating a gene in nerve cells but not muscle cells.
Fundamental research: It could accelerate our understanding of the genome by assisting in mapping its crucial functional elements and defining their roles, identifying the most essential DNA instructions for regulating a specific cell type's function.
For example, we used AlphaGenome to investigate the potential mechanism of a cancer-associated mutation. In an existing study of patients with T-cell acute lymphoblastic leukemia (T-ALL), researchers observed mutations at particular locations in the genome. Using AlphaGenome, we predicted that the mutations would activate a nearby gene called TAL1 by introducing a MYB DNA binding motif, which replicated the known disease mechanism and highlighted AlphaGenome’s ability to link specific non-coding variants to disease genes.

“
AlphaGenome will be a powerful tool for the field. Determining the relevance of different non-coding variants can be extremely challenging, particularly to do at scale. This tool will provide a crucial piece of the puzzle, allowing us to make better connections to understand diseases like cancer.

Professor Marc Mansour, University College London

Current limitations
AlphaGenome marks a significant step forward, but it's important to acknowledge its current limitations.

Like other sequence-based models, accurately capturing the influence of very distant regulatory elements, like those over 100,000 DNA letters away, is still an ongoing challenge. Another priority for future work is further increasing the model’s ability to capture cell- and tissue-specific patterns.

We haven't designed or validated AlphaGenome for personal genome prediction, a known challenge for AI models. Instead, we focused more on characterising the performance on individual genetic variants. And while AlphaGenome can predict molecular outcomes, it doesn't give the full picture of how genetic variations lead to complex traits or diseases. These often involve broader biological processes, like developmental and environmental factors, that are beyond the direct scope of our model.

We’re continuing to improve our models and gathering feedback to help us address these gaps.
"""

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
            if event.content and event.content.parts and event.content.parts[0].text:
                print(f"----->>> [Evento] Autor: {event.author}, Tipo: {type(event).__name__}, Contenido: '{event.content.parts[0].text}'")

            if event.is_final_response():
                if event.content and event.content.parts:
                    default_final_response = event.content.parts[0].text
                
        print(f"AI response: {default_final_response}")

        GEMINI_MODEL = "gemini-2.5-flash"

        extractor_agent = LlmAgent(
        name="ExtractorAgent",
        model=GEMINI_MODEL,
        instruction="""Eres un Especialista en Extracción de Información de Documentos.
        Basándote *únicamente* en el documento proporcionado por el usuario, extrae la información más importante y relevante.

        **Tu tarea:**
        1. Identifica los puntos clave, datos importantes, fechas, nombres, cifras, y conceptos principales
        2. Organiza la información extraída de manera clara y estructurada
        3. Mantén la objetividad y no agregues interpretaciones personales

        **Formato de salida:**
        Presenta la información extraída en categorías claras como:
        - Información General
        - Datos Numéricos/Estadísticas  
        - Personas/Entidades Mencionadas
        - Fechas Importantes
        - Puntos Clave del Contenido

        Proporciona *solo* la información extraída de forma organizada, sin comentarios adicionales.""",
        
        description="Extrae información clave y datos importantes de documentos.",
        output_key="informacion_extraida"
        )

        analizador_agent = LlmAgent(
        name="AnalizadorAgent", 
        model=GEMINI_MODEL,
        instruction="""Eres un Analista Experto de Contenido.
        Tu tarea es analizar profundamente la información extraída y generar insights valiosos.

        **Información a Analizar:**
        {informacion_extraida}

        **Criterios de Análisis:**
        1. **Tendencias y Patrones:** ¿Qué tendencias o patrones emergen de los datos?
        2. **Implicaciones:** ¿Cuáles son las implicaciones más importantes de esta información?
        3. **Relaciones:** ¿Cómo se relacionan entre sí los diferentes elementos?
        4. **Oportunidades:** ¿Qué oportunidades o riesgos se pueden identificar?
        5. **Context:** ¿Qué significa esta información en el contexto más amplio?

        **Salida:**
        Proporciona un análisis estructurado con insights claros y accionables.
        Enfócate en los hallazgos más significativos que agreguen valor al documento original.
        Presenta *solo* el análisis sin comentarios adicionales.""",
        description="Analiza información extraída para generar insights valiosos.",
        output_key="analisis_insights"
        )

        creador_linkedin_agent = LlmAgent(
        name="CreadorLinkedInAgent",
        model=GEMINI_MODEL,
        instruction="""Eres un Especialista en Marketing de Contenido para LinkedIn.
        Tu objetivo es crear posts atractivos y profesionales para LinkedIn basados en la información y análisis proporcionados.

        **Información Extraída:**
        {informacion_extraida}

        **Análisis e Insights:**
        {analisis_insights}

        **Tarea:**
        Crea un post para LinkedIn que sea:
        1. **Atractivo:** Que capture la atención desde las primeras líneas
        2. **Profesional:** Mantén un tono apropiado para la red profesional
        3. **Valioso:** Que aporte insights útiles a la audiencia
        4. **Accionable:** Que incluya llamadas a la acción relevantes
        5. **Engaging:** Que fomente la interacción y comentarios

        **Estructura del Post:**
        - Hook inicial impactante (1-2 líneas)
        - Desarrollo del insight principal (3-4 párrafos cortos)
        - 3-5 puntos clave con emojis apropiados
        - Llamada a la acción final
        - Hashtags relevantes (5-8 hashtags)

        **Estilo:**
        - Usa párrafos cortos para facilitar lectura
        - Incluye emojis estratégicamente
        - Mantén un tono conversacional pero profesional
        - Longitud ideal: 200-300 palabras

        **Salida:**
        Proporciona *únicamente* el post final listo para publicar en LinkedIn.""",
        description="Crea posts atractivos para LinkedIn basados en extracción y análisis de documentos.",
        output_key="post_linkedin"
        )

        pipeline_agent = SequentialAgent(
            name="pipelineAgent",
            sub_agents=[extractor_agent,analizador_agent,creador_linkedin_agent],
            description="Ejecuta una secuencia de extracción, analisis y creación de posts para linkedin basado en documentos"
        )

        APP_NAME = "pipeline_agent_documents"
        USER_ID = "user_1"
        SESSION_ID = "session_001"

        session_service = InMemorySessionService()

        session = await session_service.create_session(
            user_id=USER_ID,
            session_id=SESSION_ID,
            app_name=APP_NAME
        )

        runner = Runner(
            agent=pipeline_agent,
            session_service= session_service,
            session_id = SESSION_ID
        )

        await call_agent_async(
            texto_entrada,
            runner=runner,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
