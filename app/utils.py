"""
Funciones utilitarias para el sistema RAG.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional


def setup_logging() -> logging.Logger:
    """
    Configura el sistema de logging.

    Returns:
        logging.Logger: Logger configurado
    """
    # Crear logger
    logger = logging.getLogger("rag_medico")
    logger.setLevel(logging.INFO)

    # Crear handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Crear formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Añadir handler al logger
    logger.addHandler(console_handler)

    return logger


def initialize_directories() -> None:
    """
    Crea los directorios necesarios para el sistema.
    """
    # Directorios principales
    directories = [
        "storage",
        "storage/chromadb",
        "dataset",
        "dataset/processed_data",
        "dataset/raw_data",
        "logs",
    ]

    # Crear directorios si no existen
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_results_to_json(results: Dict[str, Any], filename: str) -> None:
    """
    Guarda resultados en formato JSON.

    Args:
        results (Dict[str, Any]): Resultados a guardar
        filename (str): Nombre del archivo
    """
    # Asegurar que existe el directorio
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Guardar resultados
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_results_from_json(filename: str) -> Optional[Dict[str, Any]]:
    """
    Carga resultados desde un archivo JSON.

    Args:
        filename (str): Nombre del archivo

    Returns:
        Optional[Dict[str, Any]]: Resultados cargados o None si hay error
    """
    if not os.path.exists(filename):
        return None

    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error al cargar resultados: {e}")
        return None


def format_retrieved_info(documents: List[Dict[str, Any]]) -> str:
    """
    Formatea la información recuperada para presentarla al usuario.

    Args:
        documents (List[Dict[str, Any]]): Documentos recuperados

    Returns:
        str: Información formateada
    """
    formatted_info = "\n"

    for i, doc in enumerate(documents, 1):
        formatted_info += f"Documento {i}:\n"
        formatted_info += f"  Contenido: {doc['content'][:200]}...\n"
        formatted_info += f"  Fuente: {doc.get('source', 'No disponible')}\n\n"

    return formatted_info
