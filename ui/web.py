"""
Interfaz web con Gradio para el sistema RAG médico.
"""
import os
import time
import gradio as gr
from typing import Dict, Any, List, Tuple, Callable
import logging

from config.settings import Settings
from app.main import answer_query


def create_web_interface(
    rag_components: Dict[str, Any], 
    settings: Settings
) -> Callable:
    """
    Crea la función de inferencia para la interfaz web.
    
    Args:
        rag_components (Dict[str, Any]): Componentes del sistema RAG
        settings (Settings): Configuración del sistema
        
    Returns:
        Callable: Función de inferencia para Gradio
    """
    logger = logging.getLogger("rag_medico")
    
    def inference_fn(query: str, history: List[Tuple[str, str]]) -> str:
        """
        Función de inferencia para Gradio.
        
        Args:
            query (str): Consulta del usuario
            history (List[Tuple[str, str]]): Historial de conversación
            
        Returns:
            str: Respuesta generada
        """
        if not query.strip():
            return "Por favor, ingrese una consulta médica."
            
        try:
            logger.info(f"Consulta recibida: {query}")
            
            # Medir tiempo
            start_time = time.time()
            
            # Generar respuesta
            answer = answer_query(query, rag_components, settings)
            
            # Registrar tiempo
            total_time = time.time() - start_time
            logger.info(f"Respuesta generada en {total_time:.2f} segundos")
            
            return answer
            
        except Exception as e:
            logger.exception("Error en la interfaz web")
            return f"Lo siento, ocurrió un error: {str(e)}"
    
    return inference_fn


def run_web_interface(
    rag_components: Dict[str, Any], 
    settings: Settings,
    share: bool = False
) -> None:
    """
    Ejecuta la interfaz web con Gradio.
    
    Args:
        rag_components (Dict[str, Any]): Componentes del sistema RAG
        settings (Settings): Configuración del sistema
        share (bool): Si se debe compartir la interfaz públicamente
    """
    # Crear función de inferencia
    inference_fn = create_web_interface(rag_components, settings)
    
    # Configurar tema
    theme = gr.Theme(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="gray"
    )
    
    # Crear interfaz de chat
    demo = gr.ChatInterface(
        fn=inference_fn,
        title="Asistente Médico RAG",
        description="""
        Este asistente médico utiliza Retrieval Augmented Generation (RAG) para proporcionar 
        información médica basada en fuentes confiables. 

        **Modelo LLM:** {llm_model}  
        **Modelo de embeddings:** {embedding_model}

        *Nota: Este asistente es solo para fines informativos y no reemplaza la consulta 
        con un profesional de la salud.*
        """.format(
            llm_model=settings.llm_model,
            embedding_model=settings.embedding_model
        ),
        examples=settings.examples,
        retry_btn="Reintentar",
        undo_btn="Deshacer",
        clear_btn="Limpiar",
        theme=theme
    )
    
    # Ejecutar la interfaz
    demo.launch(
        server_name="0.0.0.0",  # Accesible desde cualquier IP
        share=share,            # Compartir públicamente si es True
        inbrowser=True,         # Abrir en el navegador
        show_api=False          # No mostrar API pública
    )