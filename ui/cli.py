"""
Interfaz de línea de comandos para el sistema RAG médico.
"""
import sys
import time
from typing import Dict, Any, List, Optional
import logging

from config.settings import Settings
from app.main import answer_query


def print_welcome():
    """Imprime mensaje de bienvenida."""
    print("\n" + "="*50)
    print("    RAG MÉDICO - ASISTENTE DE CONSULTAS MÉDICAS")
    print("="*50)
    print("\nEste sistema utiliza Retrieval Augmented Generation (RAG)")
    print("para proporcionar respuestas a consultas médicas basadas")
    print("en información confiable de fuentes médicas.")
    print("\nEscriba 'salir', 'exit' o 'q' para terminar.")
    print("Escriba 'ayuda' o 'help' o '?' para ver comandos disponibles.")
    print("="*50 + "\n")


def print_help():
    """Imprime ayuda con comandos disponibles."""
    print("\nComandos disponibles:")
    print("  salir, exit, q       - Salir del programa")
    print("  ayuda, help, ?       - Mostrar esta ayuda")
    print("  fuentes, sources     - Mostrar las fuentes de la última respuesta")
    print("  debug                - Alternar modo de debug (muestra información adicional)")
    print("  clear, cls           - Limpiar la pantalla")
    print("  ejemplo, example [n] - Usar consulta de ejemplo número n (1-4)")
    print()


def print_sources(last_documents: Optional[List[Dict[str, Any]]]):
    """
    Muestra las fuentes de la última respuesta.
    
    Args:
        last_documents (Optional[List[Dict[str, Any]]]): Documentos recuperados
    """
    if not last_documents:
        print("\nNo hay fuentes disponibles. Realice una consulta primero.\n")
        return
        
    print("\nFuentes utilizadas para la última respuesta:")
    for i, doc in enumerate(last_documents, 1):
        print(f"\n[Fuente {i}]")
        print(f"URL: {doc.metadata.get('source', 'No disponible')}")
        print(f"Pregunta original: {doc.metadata.get('question', 'No disponible')}")
        print(f"Contenido: {doc.page_content[:150]}...")
    print()


def clear_screen():
    """Limpia la pantalla de la terminal."""
    print("\033[H\033[J", end="")


def get_example_query(examples: List[str], index: Optional[int] = None) -> str:
    """
    Obtiene una consulta de ejemplo.
    
    Args:
        examples (List[str]): Lista de consultas de ejemplo
        index (Optional[int]): Índice del ejemplo a usar (1-indexed)
        
    Returns:
        str: Consulta de ejemplo
    """
    if not examples:
        return "¿Qué es la diabetes?"
        
    if index is None or index < 1 or index > len(examples):
        # Mostrar lista de ejemplos disponibles
        print("\nConsultas de ejemplo disponibles:")
        for i, example in enumerate(examples, 1):
            print(f"  {i}. {example}")
        
        try:
            selection = int(input("\nSeleccione un número (1-{0}): ".format(len(examples))))
            if selection < 1 or selection > len(examples):
                print("Selección no válida, usando ejemplo 1.")
                return examples[0]
            return examples[selection - 1]
        except ValueError:
            print("Entrada no válida, usando ejemplo 1.")
            return examples[0]
    
    return examples[index - 1]


def run_console_interface(rag_components: Dict[str, Any], settings: Settings):
    """
    Ejecuta la interfaz de línea de comandos.
    
    Args:
        rag_components (Dict[str, Any]): Componentes del sistema RAG
        settings (Settings): Configuración del sistema
    """
    logger = logging.getLogger("rag_medico")
    
    # Estado de la interfaz
    debug_mode = False
    last_documents = None
    last_context = None
    
    # Mostrar bienvenida
    print_welcome()
    
    # Bucle principal
    while True:
        try:
            # Obtener consulta del usuario
            query = input("\n> ").strip()
            
            # Procesar comandos especiales
            if query.lower() in ['salir', 'exit', 'q']:
                print("\n¡Hasta pronto!\n")
                break
                
            elif query.lower() in ['ayuda', 'help', '?']:
                print_help()
                continue
                
            elif query.lower() in ['fuentes', 'sources']:
                print_sources(last_documents)
                continue
                
            elif query.lower() == 'debug':
                debug_mode = not debug_mode
                print(f"\nModo debug {'activado' if debug_mode else 'desactivado'}.\n")
                continue
                
            elif query.lower() in ['clear', 'cls']:
                clear_screen()
                continue
                
            elif query.lower().startswith(('ejemplo', 'example')):
                parts = query.split()
                index = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                query = get_example_query(settings.examples, index)
                print(f"\n> {query}")
            
            # Ignorar consultas vacías
            if not query:
                continue
            
            # Medir tiempo de respuesta
            start_time = time.time()
            
            # Recuperar documentos relevantes
            retriever = rag_components['retriever']
            documents, context = retriever.retrieve(query, k=settings.n_retrieval_results)
            last_documents = documents
            last_context = context
            
            retrieval_time = time.time() - start_time
            
            # Mostrar información de recuperación en modo debug
            if debug_mode:
                print(f"\n[Debug] Recuperados {len(documents)} documentos en {retrieval_time:.2f} segundos")
                print(f"\n[Debug] Contexto utilizado:\n{context}")
            
            # Generar respuesta
            print("\nGenerando respuesta...")
            answer = answer_query(query, rag_components, settings)
            
            total_time = time.time() - start_time
            
            # Mostrar respuesta
            print("\nRespuesta:")
            print(f"{answer}")
            
            if debug_mode:
                print(f"\n[Debug] Tiempo total: {total_time:.2f} segundos")
                
        except KeyboardInterrupt:
            print("\n\nOperación cancelada por el usuario.")
            print("Escriba 'salir' para terminar o continúe con una nueva consulta.")
            
        except Exception as e:
            logger.exception("Error en la interfaz de consola")
            print(f"\nOcurrió un error: {str(e)}")
            print("Por favor, intente de nuevo.")