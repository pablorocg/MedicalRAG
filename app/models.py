"""
Módulo para la gestión de modelos, embeddings y generación de texto.
"""

import json
from typing import List

import requests
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from tqdm import tqdm


class OllamaClient:
    """
    Cliente para interactuar con Ollama.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Inicializa el cliente de Ollama.

        Args:
            base_url (str): URL base del servicio Ollama
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.headers = {"Content-Type": "application/json"}

    def check_connection(self) -> bool:
        """
        Verifica si Ollama está disponible.

        Returns:
            bool: True si Ollama está disponible, False en caso contrario
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, prompt: str, model: str, temperature: float = 0.7) -> str:
        """
        Genera texto con un modelo de Ollama.

        Args:
            prompt (str): Texto de entrada
            model (str): Nombre del modelo en Ollama
            temperature (float): Temperatura para la generación

        Returns:
            str: Texto generado
        """
        url = f"{self.base_url}/api/generate"
        data = {"model": model, "prompt": prompt, "temperature": temperature}

        full_response = []
        try:
            response = self.session.post(
                url, data=json.dumps(data), headers=self.headers, stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    decoded_line = json.loads(line.decode("utf-8"))
                    full_response.append(decoded_line["response"])
        except Exception as e:
            print(f"Error en generación: {e}")
            return f"Error: No se pudo generar respuesta. {str(e)}"
        finally:
            response.close()

        return "".join(full_response)

    def get_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        """
        Obtiene embeddings para una lista de textos.

        Args:
            texts (List[str]): Lista de textos
            model (str): Modelo de embeddings

        Returns:
            List[List[float]]: Lista de embeddings
        """
        url = f"{self.base_url}/api/embeddings"
        embeddings = []

        for text in tqdm(texts, desc="Generando embeddings"):
            data = {"model": model, "prompt": text}

            try:
                response = self.session.post(
                    url, data=json.dumps(data), headers=self.headers
                )
                response.raise_for_status()
                result = response.json()
                embeddings.append(result["embedding"])
            except Exception as e:
                print(f"Error al obtener embedding: {e}")
                # Devolver un embedding de ceros en caso de error
                if embeddings:
                    embeddings.append([0.0] * len(embeddings[0]))
                else:
                    embeddings.append([0.0] * 1536)  # Dimensión típica

        return embeddings


class OllamaEmbeddingProvider:
    """
    Proveedor de embeddings usando Ollama, compatible con LangChain.
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        """
        Inicializa el proveedor de embeddings.

        Args:
            model_name (str): Nombre del modelo de embeddings
            base_url (str): URL base del servicio Ollama
        """
        # Versión directa con requests
        self.client = OllamaClient(base_url)
        self.model_name = model_name

        # Versión con LangChain
        self.lc_embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)

    def embed_documents(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """
        Genera embeddings para documentos con procesamiento por lotes.

        Args:
            texts (List[str]): Lista de textos de documentos
            batch_size (int): Tamaño del lote para procesar

        Returns:
            List[List[float]]: Lista de embeddings
        """
        # Usar implementación de LangChain
        all_embeddings = []

        # Procesar en lotes para evitar problemas de memoria
        for i in tqdm(range(0, len(texts), batch_size), desc="Generando embeddings"):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self.lc_embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Genera embedding para una consulta.

        Args:
            text (str): Texto de la consulta

        Returns:
            List[float]: Embedding de la consulta
        """
        # Usar implementación de LangChain
        return self.lc_embeddings.embed_query(text)


class OllamaTextGenerator:
    """
    Generador de texto usando Ollama, compatible con LangChain.
    """

    def __init__(
        self, model_name: str = "gemma2:2b", base_url: str = "http://localhost:11434"
    ):
        """
        Inicializa el generador de texto.

        Args:
            model_name (str): Nombre del modelo de generación
            base_url (str): URL base del servicio Ollama
        """
        # Versión directa con requests
        self.client = OllamaClient(base_url)
        self.model_name = model_name

        # Versión con LangChain
        self.lc_llm = Ollama(model=model_name, base_url=base_url)

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Genera texto a partir de un prompt.

        Args:
            prompt (str): Texto de entrada
            temperature (float): Temperatura para la generación

        Returns:
            str: Texto generado
        """
        return self.client.generate(prompt, self.model_name, temperature)

    def generate_from_langchain(self, prompt: str) -> str:
        """
        Genera texto usando la interfaz de LangChain.

        Args:
            prompt (str): Texto de entrada

        Returns:
            str: Texto generado
        """
        return self.lc_llm.invoke(prompt)


def generate_prompt(query: str, context: str) -> str:
    """
    Genera un prompt para el LLM con contexto de RAG.

    Args:
        query (str): Consulta del usuario
        context (str): Contexto recuperado

    Returns:
        str: Prompt estructurado
    """
    return f"""
        Como asistente médico especializado, proporciona información precisa y bien fundamentada 
        basándote exclusivamente en el contexto proporcionado, sin utilizar conocimiento previo.
        
        Pregunta del usuario: "{query}"
        
        Considerando únicamente esta información contextual:
        {context}
        
        Usa esta información para dar una respuesta clara, concisa y que responda directamente 
        a la pregunta del usuario. Si la información sugiere la necesidad de consultar a un 
        profesional o realizar una exploración más detallada, indícalo.
    """


if __name__ == "__main__":
    # Ejemplo de uso
    query = "¿Qué es la diabetes?"
    context = "La diabetes es una enfermedad crónica que se caracteriza por niveles elevados de glucosa en sangre."

    # Inicializar componentes
    embedding_provider = OllamaEmbeddingProvider()
    text_generator = OllamaTextGenerator()

    # Verificar la conexión antes de proceder
    if not text_generator.client.check_connection():
        print(
            "Error: No se puede conectar a Ollama. Verifique que el servicio esté en ejecución."
        )
        exit(1)

    print(
        f"Modelos disponibles: {json.loads(requests.get('http://localhost:11434/api/tags').text)['models']}"
    )

    # Generar prompt
    prompt = generate_prompt(query, context)

    # Generar respuesta
    answer = text_generator.generate(prompt)

    print(f"Respuesta generada: {answer}")

    # Generar embeddings
    documents = [
        "La diabetes es una enfermedad crónica.",
        "La diabetes tipo 2 es la más común.",
    ]
    embeddings = embedding_provider.embed_documents(documents)

    print(
        f"Embeddings generados: {len(embeddings)} embeddings con {len(embeddings[0])} dimensiones cada uno"
    )

    # Generar embedding de consulta
    query_embedding = embedding_provider.embed_query("¿Qué es la diabetes?")
    print(f"Embedding de consulta generado: {len(query_embedding)} dimensiones")
