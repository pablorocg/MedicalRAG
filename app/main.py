"""
Punto de entrada principal para el sistema RAG médico.
"""
import os
import argparse
from typing import Dict, Any

from config.settings import DEFAULT_SETTINGS, Settings
from data.dataset import get_all_data, TextDataset
from app.models import OllamaEmbeddingProvider, OllamaTextGenerator, generate_prompt
from app.retriever import DocumentProcessor, ChromaVectorStore, RAGRetriever
from app.utils import setup_logging, initialize_directories

# Configurar logging
logger = setup_logging()


def parse_arguments():
    """
    Analiza los argumentos de línea de comandos.
    
    Returns:
        argparse.Namespace: Argumentos analizados
    """
    parser = argparse.ArgumentParser(description='RAG Médico con Ollama')
    
    # Modos de ejecución
    parser.add_argument('--mode', choices=['console', 'web', 'evaluate'], default='console',
                        help='Modo de ejecución: console, web o evaluate')
    
    # Modelos y configuración
    parser.add_argument('--llm_model', type=str, default=DEFAULT_SETTINGS.llm_model,
                        help='Modelo LLM en Ollama')
    parser.add_argument('--embedding_model', type=str, default=DEFAULT_SETTINGS.embedding_model,
                        help='Modelo de embeddings en Ollama')
    parser.add_argument('--n_retrieval_results', type=int, default=DEFAULT_SETTINGS.n_retrieval_results,
                        help='Número de documentos a recuperar')
    
    # Procesamiento
    parser.add_argument('--batch_size', type=int, default=DEFAULT_SETTINGS.batch_size,
                        help='Tamaño del batch para procesamiento')
    parser.add_argument('--chunk_size', type=int, default=DEFAULT_SETTINGS.chunk_size,
                        help='Tamaño del chunk para división de texto')
    parser.add_argument('--chunk_overlap', type=int, default=DEFAULT_SETTINGS.chunk_overlap,
                        help='Solapamiento entre chunks')
    
    # Entorno
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=DEFAULT_SETTINGS.device,
                        help='Dispositivo para procesamiento')
    parser.add_argument('--ollama_base_url', type=str, default=DEFAULT_SETTINGS.ollama_base_url,
                        help='URL base de Ollama')
    
    # Otras opciones
    parser.add_argument('--rebuild_index', action='store_true',
                      help='Reconstruir el índice de vectores desde cero')
    parser.add_argument('--max_documents', type=int, default=None,
                      help='Número máximo de documentos a procesar (para pruebas)')
    parser.add_argument('--larger_chunks', action='store_true',
                      help='Usar chunks más grandes para reducir el número total')
    
    return parser.parse_args()


def setup_rag_system(settings: Settings, rebuild_index: bool = False, args: argparse.Namespace = None) -> Dict[str, Any]:
    """
    Configura el sistema RAG.
    
    Args:
        settings (Settings): Configuración del sistema
        rebuild_index (bool): Si se debe reconstruir el índice
        
    Returns:
        Dict[str, Any]: Componentes del sistema RAG
    """
    # Inicializar directorios
    initialize_directories()
    
    # Inicializar proveedores
    embedding_provider = OllamaEmbeddingProvider(
        model_name=settings.embedding_model,
        base_url=settings.ollama_base_url
    )
    
    text_generator = OllamaTextGenerator(
        model_name=settings.llm_model,
        base_url=settings.ollama_base_url
    )
    
    # Inicializar vector store
    vector_store = ChromaVectorStore(
        embedding_provider=embedding_provider,
        persist_directory=settings.chroma_db_path,
        collection_name=settings.collection_name
    )
    
    # Intentar cargar el índice existente
    index_loaded = False
    if not rebuild_index:
        index_loaded = vector_store.load()
        if index_loaded:
            logger.info("Índice vectorial cargado correctamente")
        else:
            logger.warning("No se pudo cargar el índice existente, se creará uno nuevo")
    
    # Si no se pudo cargar o se solicitó reconstruir, crear nuevo índice
    if rebuild_index or not index_loaded:
        logger.info("Creando nuevo índice vectorial")
        
        try:
            # Cargar dataset
            logger.info("Cargando dataset")
            df = get_all_data(os.path.join("dataset", "processed_data"))
            
            if df.empty:
                logger.error("No se pudieron cargar datos. Verifica que existan archivos en dataset/processed_data/")
                raise ValueError("No hay datos disponibles para crear el índice")
                
            dataset = TextDataset(df)
            
            # Procesar documentos
            logger.info("Procesando documentos")
            
            # Ajustar tamaño de chunk según preferencia
            chunk_size = 1024 if settings.larger_chunks else settings.chunk_size
            chunk_overlap = 128 if settings.larger_chunks else settings.chunk_overlap
            
            doc_processor = DocumentProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Usar max_documents si se especificó
            documents = doc_processor.create_langchain_documents(
                dataset, 
                max_samples=args.max_documents
            )
            
            if not documents:
                logger.error("No se pudieron procesar documentos.")
                raise ValueError("No hay documentos disponibles para crear el índice")
            
            # Crear índice con procesamiento por lotes
            vector_store.create_from_documents(documents, batch_size=args.batch_size)
            
            # Verificar que se creó correctamente
            if not vector_store.vectordb:
                logger.error("No se pudo crear el índice vectorial.")
                raise ValueError("Error al crear índice vectorial")
                
        except Exception as e:
            logger.exception(f"Error al crear el índice: {e}")
            # Usar algún índice de muestra o datos predeterminados para pruebas
            logger.info("Usando dataset de muestra para pruebas")
            
            # Crear datos de ejemplo (solo para pruebas)
            from langchain_core.documents import Document
            sample_docs = [
                Document(page_content="La diabetes es una enfermedad crónica que aparece cuando el páncreas no produce suficiente insulina.", 
                         metadata={"question": "¿Qué es la diabetes?", "source": "ejemplo.com"}),
                Document(page_content="Los síntomas de la diabetes incluyen sed excesiva, micción frecuente y fatiga.", 
                         metadata={"question": "¿Cuáles son los síntomas de la diabetes?", "source": "ejemplo.com"}),
                Document(page_content="La enfermedad celíaca es una afección autoinmune que daña el intestino delgado cuando se ingiere gluten.", 
                         metadata={"question": "¿Qué es la enfermedad celíaca?", "source": "ejemplo.com"})
            ]
            vector_store.create_from_documents(sample_docs)
    
    # Verificar que el vector_store esté inicializado correctamente
    if not vector_store.vectordb:
        logger.error("La base de datos vectorial no está inicializada correctamente.")
        raise RuntimeError("Error en la configuración del sistema RAG: base de datos vectorial no inicializada")
        
    # Inicializar recuperador
    retriever = RAGRetriever(vector_store)
    
    return {
        'embedding_provider': embedding_provider,
        'text_generator': text_generator,
        'vector_store': vector_store,
        'retriever': retriever
    }


def answer_query(query: str, rag_components: Dict[str, Any], settings: Settings) -> str:
    """
    Responde a una consulta utilizando RAG.
    
    Args:
        query (str): Consulta del usuario
        rag_components (Dict[str, Any]): Componentes del sistema RAG
        settings (Settings): Configuración del sistema
        
    Returns:
        str: Respuesta generada
    """
    # Recuperar documentos relevantes
    retriever = rag_components['retriever']
    text_generator = rag_components['text_generator']
    
    # Obtener contexto
    documents, context = retriever.retrieve(query, k=settings.n_retrieval_results)
    
    # Generar prompt
    prompt = generate_prompt(query, context)
    
    # Generar respuesta
    answer = text_generator.generate(prompt)
    
    return answer


def main():
    """Función principal del sistema RAG."""
    # Parsear argumentos
    args = parse_arguments()
    
    # Configurar sistema
    settings = Settings.from_args(args)
    logger.info(f"Configuración: {vars(settings)}")
    
    # Inicializar sistema RAG
    rag_components = setup_rag_system(settings, rebuild_index=args.rebuild_index)
    
    # Ejecutar en modo apropiado
    if args.mode == 'console':
        from ui.cli import run_console_interface
        run_console_interface(rag_components, settings)
    elif args.mode == 'web':
        from ui.web import run_web_interface
        run_web_interface(rag_components, settings)
    elif args.mode == 'evaluate':
        from app.evaluation import run_evaluation
        run_evaluation(rag_components, settings)
    else:
        logger.error(f"Modo no válido: {args.mode}")


if __name__ == "__main__":
    main()