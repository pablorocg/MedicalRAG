"""
Módulo para la recuperación de documentos relevantes utilizando ChromaDB.
"""

import os
from typing import Any, Dict, List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm

from app.models import OllamaEmbeddingProvider
from data.dataset import TextDataset


class DocumentProcessor:
    """
    Procesador de documentos para dividir y preparar textos para embeddings.
    """

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128):
        """
        Inicializa el procesador de documentos.

        Args:
            chunk_size (int): Tamaño de cada chunk de texto
            chunk_overlap (int): Solapamiento entre chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def prepare_documents(
        self, dataset: TextDataset, max_samples: int = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Prepara documentos para indexación, dividiendo textos largos en chunks.

        Args:
            dataset (TextDataset): Dataset con preguntas y respuestas
            max_samples (int, optional): Número máximo de muestras a procesar

        Returns:
            Tuple[List[str], List[Dict[str, Any]]]: Tuple con (textos, metadatos)
        """
        texts = []
        metadatas = []

        # Limitar muestras si es necesario
        if max_samples and max_samples < len(dataset):
            process_range = range(max_samples)
            print(
                f"Procesando un subconjunto de {max_samples} documentos de {len(dataset)}"
            )
        else:
            process_range = range(len(dataset))

        for idx in tqdm(process_range, desc="Preparando documentos"):
            item = dataset[idx]

            # Crear documento combinado (solo respuesta para reducir tamaño)
            # Esto reduce significativamente la cantidad de chunks
            combined_text = f"{item['answer']}"

            # Solo dividir en chunks textos muy largos (> 2048 caracteres)
            if len(combined_text) > 2048:
                chunks = self.text_splitter.split_text(combined_text)
                for chunk in chunks:
                    texts.append(chunk)
                    metadatas.append(
                        {
                            "question": item["question"],
                            "source": item["url"],
                            "chunk": True,
                        }
                    )
            else:
                texts.append(combined_text)
                metadatas.append(
                    {
                        "question": item["question"],
                        "source": item["url"],
                        "chunk": False,
                    }
                )

        print(f"Documentos preparados: {len(texts)} chunks de texto")
        return texts, metadatas

    def create_langchain_documents(
        self, dataset: TextDataset, max_samples: int = None
    ) -> List[Document]:
        """
        Crea documentos en formato LangChain a partir del dataset.

        Args:
            dataset (TextDataset): Dataset con preguntas y respuestas
            max_samples (int, optional): Número máximo de muestras a procesar

        Returns:
            List[Document]: Lista de documentos LangChain
        """
        texts, metadatas = self.prepare_documents(dataset, max_samples)
        documents = []

        for text, metadata in zip(texts, metadatas):
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)

        return documents


class ChromaVectorStore:
    """
    Almacén de vectores usando ChromaDB para RAG.
    """

    def __init__(
        self,
        embedding_provider: OllamaEmbeddingProvider,
        persist_directory: str,
        collection_name: str = "medical_rag",
    ):
        """
        Inicializa el almacén de vectores.

        Args:
            embedding_provider (OllamaEmbeddingProvider): Proveedor de embeddings
            persist_directory (str): Directorio para persistir la base de datos
            collection_name (str): Nombre de la colección
        """
        self.embedding_provider = embedding_provider
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectordb = None

    def create_from_documents(
        self, documents: List[Document], batch_size: int = 128
    ) -> None:
        """
        Crea el almacén de vectores a partir de documentos con procesamiento por lotes.

        Args:
            documents (List[Document]): Lista de documentos LangChain
            batch_size (int): Tamaño del lote para procesamiento
        """
        print(f"Creando base de datos vectorial con {len(documents)} documentos...")

        # Si tenemos muchos documentos, procesamos por lotes
        if len(documents) > batch_size:
            from langchain_community.vectorstores import Chroma

            # Procesar en lotes
            for i in tqdm(
                range(0, len(documents), batch_size),
                desc="Procesando lotes de documentos",
            ):
                batch_documents = documents[i : i + batch_size]

                # Para el primer lote, crear la base de datos
                if i == 0:
                    self.vectordb = Chroma.from_documents(
                        documents=batch_documents,
                        embedding=self.embedding_provider.lc_embeddings,
                        persist_directory=self.persist_directory,
                        collection_name=self.collection_name,
                    )
                # Para los lotes siguientes, añadir documentos a la base existente
                else:
                    self.vectordb.add_documents(documents=batch_documents)

                # Persistir después de cada lote para evitar pérdida de datos
                if self.vectordb:
                    self.vectordb.persist()
        else:
            # Si hay pocos documentos, usar el método estándar
            from langchain_community.vectorstores import Chroma

            self.vectordb = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_provider.lc_embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
            )

            # Persistir la base de datos
            self.vectordb.persist()

        print(
            f"Base de datos vectorial creada y persistida en {self.persist_directory}"
        )

    def load(self) -> bool:
        """
        Carga la base de datos vectorial existente.

        Returns:
            bool: True si se cargó correctamente, False en caso contrario
        """
        if not os.path.exists(self.persist_directory):
            print(
                f"No se encontró una base de datos vectorial en {self.persist_directory}"
            )
            return False

        try:
            # Primero verificamos que existan los archivos necesarios
            chroma_files = [
                os.path.join(self.persist_directory, "chroma.sqlite3"),
                os.path.join(self.persist_directory, "chroma-embeddings.parquet"),
            ]

            # Verificamos si al menos el archivo principal existe
            if not os.path.exists(chroma_files[0]):
                print(
                    f"No se encontraron archivos de ChromaDB en {self.persist_directory}"
                )
                return False

            # Usamos try/except específicamente para la carga
            try:
                from langchain_community.vectorstores import Chroma

                self.vectordb = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_provider.lc_embeddings,
                    collection_name=self.collection_name,
                )
                # Verificamos que la base de datos tenga elementos
                if self.vectordb._collection.count() == 0:
                    print("La base de datos vectorial existe pero está vacía.")
                    self.vectordb = None
                    return False

                print(f"Base de datos vectorial cargada desde {self.persist_directory}")
                return True
            except Exception as e:
                print(f"Error al cargar la base de datos vectorial: {e}")
                self.vectordb = None
                return False
        except Exception as e:
            print(f"Error al verificar la base de datos vectorial: {e}")
            return False

    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Busca documentos relevantes para una consulta.

        Args:
            query (str): Texto de la consulta
            k (int): Número de documentos a recuperar

        Returns:
            List[Document]: Lista de documentos relevantes
        """
        if not self.vectordb:
            raise ValueError("La base de datos vectorial no está inicializada")

        return self.vectordb.similarity_search(query, k=k)


class RAGRetriever:
    """
    Recuperador de información para RAG (Retrieval Augmented Generation).
    """

    def __init__(self, vector_store: ChromaVectorStore):
        """
        Inicializa el recuperador.

        Args:
            vector_store (ChromaVectorStore): Almacén de vectores
        """
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[Document], str]:
        """
        Recupera documentos relevantes y genera el contexto formateado.

        Args:
            query (str): Consulta del usuario
            k (int): Número de documentos a recuperar

        Returns:
            Tuple[List[Document], str]: Documentos recuperados y contexto formateado
        """
        documents = self.vector_store.search(query, k=k)
        context = self._format_retrieved_context(documents)
        return documents, context

    def _format_retrieved_context(self, documents: List[Document]) -> str:
        """
        Formatea los documentos recuperados en un contexto legible.

        Args:
            documents (List[Document]): Lista de documentos recuperados

        Returns:
            str: Contexto formateado
        """
        formatted_context = ""

        for i, doc in enumerate(documents, 1):
            formatted_context += f"=== Información {i} ===\n"
            formatted_context += f"{doc.page_content}\n"
            formatted_context += (
                f"Fuente: {doc.metadata.get('source', 'No disponible')}\n\n"
            )

        return formatted_context
