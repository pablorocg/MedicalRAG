"""
Configuraciones globales para el sistema RAG médico.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

# Rutas base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PROCESSED_DATA_DIR = os.path.join(DATASET_DIR, "processed_data")
RAW_DATA_DIR = os.path.join(DATASET_DIR, "raw_data")
CHROMA_DIR = os.path.join(STORAGE_DIR, "chromadb")

# Asegurar que los directorios existen
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

@dataclass
class Settings:
    """Configuración global para el sistema RAG."""
    # Modelos
    llm_model: str = "gemma2:2b"           # Modelo de LLM en Ollama
    embedding_model: str = "nomic-embed-text"  # Modelo de embeddings en Ollama
    
    # Parámetros de recuperación
    n_retrieval_results: int = 5          # Número de documentos a recuperar
    
    # Procesamiento
    batch_size: int = 128                 # Tamaño del batch para procesamiento
    chunk_size: int = 512                 # Tamaño del chunk para división de texto
    chunk_overlap: int = 50               # Solapamiento entre chunks
    larger_chunks: bool = False           # Usar chunks más grandes (reduce el número)
    
    # Entorno
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    ollama_base_url: str = "http://localhost:11434"
    
    # Rutas
    chroma_db_path: str = CHROMA_DIR
    collection_name: str = "medical_rag"
    
    # Interfaz
    examples: List[str] = field(default_factory=lambda: [
        "What is diabetes?", 
        "Is ginseng good for diabetes?", 
        "What are the symptoms of diabetes?", 
        "What is Celiac disease?"
    ])
    
    # Evaluación
    max_test_samples: int = 50
    
    @classmethod
    def from_args(cls, args):
        """Crea una instancia de configuración a partir de argumentos CLI."""
        settings = cls()
        # Actualizar configuraciones desde args
        for key, value in vars(args).items():
            if hasattr(settings, key) and value is not None:
                setattr(settings, key, value)
        return settings

# Instancia por defecto
DEFAULT_SETTINGS = Settings()