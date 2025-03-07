"""
Módulo para la evaluación del sistema RAG.
"""
import os
import time
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk

from config.settings import Settings
from data.dataset import load_test_dataset
from app.utils import save_results_to_json


def initialize_nltk():
    """Inicializa las dependencias de NLTK."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)


def calculate_retrieval_metrics(query_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calcula métricas relacionadas con la recuperación de documentos.
    
    Args:
        query_results (List[Dict[str, Any]]): Resultados de las consultas
        
    Returns:
        Dict[str, float]: Métricas de recuperación
    """
    if not query_results:
        return {
            'avg_retrieval_time': 0.0,
            'avg_num_docs': 0.0
        }
    
    retrieval_times = [r.get('retrieval_time', 0.0) for r in query_results]
    num_docs = [len(r.get('retrieved_docs', [])) for r in query_results]
    
    return {
        'avg_retrieval_time': np.mean(retrieval_times),
        'avg_num_docs': np.mean(num_docs)
    }


def calculate_generation_metrics(query_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calcula métricas relacionadas con la generación de respuestas.
    
    Args:
        query_results (List[Dict[str, Any]]): Resultados de las consultas
        
    Returns:
        Dict[str, float]: Métricas de generación
    """
    if not query_results:
        return {
            'avg_generation_time': 0.0,
            'avg_answer_length': 0.0
        }
    
    generation_times = [r.get('generation_time', 0.0) for r in query_results]
    answer_lengths = [len(r.get('generated_answer', '').split()) for r in query_results]
    
    return {
        'avg_generation_time': np.mean(generation_times),
        'avg_answer_length': np.mean(answer_lengths)
    }


def calculate_content_metrics(generated_answers: List[str], reference_answers: List[str]) -> Dict[str, float]:
    """
    Calcula métricas relacionadas con el contenido de las respuestas.
    
    Args:
        generated_answers (List[str]): Respuestas generadas
        reference_answers (List[str]): Respuestas de referencia
        
    Returns:
        Dict[str, float]: Métricas de contenido
    """
    # Inicializar NLTK si es necesario
    initialize_nltk()
    
    # Calcular ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, gen) for ref, gen in zip(reference_answers, generated_answers)]
    
    # Calcular BLEU
    try:
        bleu_scores = [
            sentence_bleu([word_tokenize(ref)], word_tokenize(gen))
            for ref, gen in zip(reference_answers, generated_answers)
        ]
    except Exception as e:
        print(f"Error calculando BLEU: {e}")
        bleu_scores = [0.0] * len(generated_answers)
    
    # Calcular promedios
    avg_rouge1_p = np.mean([score['rouge1'].precision for score in rouge_scores])
    avg_rouge1_r = np.mean([score['rouge1'].recall for score in rouge_scores])
    avg_rouge1_f1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
    
    avg_rouge2_f1 = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
    avg_rougeL_f1 = np.mean([score['rougeL'].fmeasure for score in rouge_scores])
    
    avg_bleu = np.mean(bleu_scores)
    
    return {
        'rouge1_precision': float(avg_rouge1_p),
        'rouge1_recall': float(avg_rouge1_r),
        'rouge1_f1': float(avg_rouge1_f1),
        'rouge2_f1': float(avg_rouge2_f1),
        'rougeL_f1': float(avg_rougeL_f1),
        'bleu': float(avg_bleu)
    }


def evaluate_single_query(
    query: str,
    reference_answer: str,
    rag_components: Dict[str, Any],
    settings: Settings
) -> Dict[str, Any]:
    """
    Evalúa una única consulta.
    
    Args:
        query (str): Consulta a evaluar
        reference_answer (str): Respuesta de referencia
        rag_components (Dict[str, Any]): Componentes del sistema RAG
        settings (Settings): Configuración del sistema
        
    Returns:
        Dict[str, Any]: Resultados de la evaluación
    """
    from app.main import answer_query
    
    # Medir tiempo de recuperación
    start_retrieval = time.time()
    retriever = rag_components['retriever']
    documents, context = retriever.retrieve(query, k=settings.n_retrieval_results)
    retrieval_time = time.time() - start_retrieval
    
    # Extraer información de documentos
    retrieved_docs = [
        {
            'content': doc.page_content,
            'metadata': doc.metadata
        }
        for doc in documents
    ]
    
    # Medir tiempo de generación
    start_generation = time.time()
    generated_answer = answer_query(query, rag_components, settings)
    generation_time = time.time() - start_generation
    
    # Compilar resultados
    result = {
        'query': query,
        'reference_answer': reference_answer,
        'generated_answer': generated_answer,
        'retrieved_docs': retrieved_docs,
        'retrieval_time': retrieval_time,
        'generation_time': generation_time,
        'total_time': retrieval_time + generation_time
    }
    
    return result


def run_evaluation(rag_components: Dict[str, Any], settings: Settings) -> None:
    """
    Ejecuta la evaluación completa del sistema.
    
    Args:
        rag_components (Dict[str, Any]): Componentes del sistema RAG
        settings (Settings): Configuración del sistema
    """
    print("\n=== Iniciando evaluación del sistema RAG ===\n")
    
    # Cargar dataset de prueba
    test_file = os.path.join("dataset", "QA-TestSet-LiveQA-Med-Qrels-2479-Answers", 
                            "All-2479-Answers-retrieved-from-MedQuAD.csv")
    
    questions, reference_answers = load_test_dataset(test_file)
    
    if not questions:
        print("No se pudo cargar el dataset de prueba.")
        return
    
    # Limitar número de muestras para evaluación
    max_samples = min(settings.max_test_samples, len(questions))
    questions = questions[:max_samples]
    reference_answers = reference_answers[:max_samples]
    
    print(f"Evaluando {max_samples} consultas...")
    
    # Evaluar cada consulta
    query_results = []
    for i, (query, reference) in enumerate(tqdm(zip(questions, reference_answers), total=max_samples)):
        result = evaluate_single_query(query, reference, rag_components, settings)
        query_results.append(result)
    
    # Extraer respuestas generadas
    generated_answers = [r.get('generated_answer', '') for r in query_results]
    
    # Calcular métricas
    retrieval_metrics = calculate_retrieval_metrics(query_results)
    generation_metrics = calculate_generation_metrics(query_results)
    content_metrics = calculate_content_metrics(generated_answers, reference_answers)
    
    # Combinar todas las métricas
    all_metrics = {
        **retrieval_metrics,
        **generation_metrics,
        **content_metrics,
        'num_queries': len(query_results),
        'llm_model': settings.llm_model,
        'embedding_model': settings.embedding_model,
        'n_retrieval_results': settings.n_retrieval_results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Guardar resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join("storage", "evaluations", f"eval_{timestamp}.json")
    save_results_to_json({
        'metrics': all_metrics,
        'query_results': query_results
    }, results_file)
    
    # Mostrar resultados
    print("\n=== Resultados de la evaluación ===\n")
    print(f"Modelo LLM: {settings.llm_model}")
    print(f"Modelo de embeddings: {settings.embedding_model}")
    print(f"Número de consultas evaluadas: {len(query_results)}")
    print("\nMétricas de recuperación:")
    print(f"  Tiempo promedio de recuperación: {retrieval_metrics['avg_retrieval_time']:.4f} s")
    print(f"  Promedio de documentos recuperados: {retrieval_metrics['avg_num_docs']:.2f}")
    
    print("\nMétricas de generación:")
    print(f"  Tiempo promedio de generación: {generation_metrics['avg_generation_time']:.4f} s")
    print(f"  Longitud promedio de respuesta: {generation_metrics['avg_answer_length']:.2f} palabras")
    
    print("\nMétricas de contenido:")
    print(f"  ROUGE-1 F1: {content_metrics['rouge1_f1']:.4f}")
    print(f"  ROUGE-2 F1: {content_metrics['rouge2_f1']:.4f}")
    print(f"  ROUGE-L F1: {content_metrics['rougeL_f1']:.4f}")
    print(f"  BLEU: {content_metrics['bleu']:.4f}")
    
    print(f"\nResultados detallados guardados en: {results_file}")