
import os
from faiss import write_index
import gradio as gr
import numpy as np
# import pandas as pd
# import requests

import argparse
from tabulate import tabulate
from tqdm import tqdm

from models import *
from utils import *


# Set environment variable to avoid MKL errors
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def answer_query(query_text, index, documents, llm_model, embedding_model, n_docs, device):
    """
    Answers a query by searching for the most similar documents using an index.

    Args:
        query_text (str): The text of the query.
        index: The index used for searching the documents.
        documents: The collection of documents.

    Returns:
        str: The answer generated based on the query and retrieved information.
    """
    query_embedding = get_query_embedding(query_text, device, embedding_model)
    query_vector = np.expand_dims(query_embedding, axis=0)
    D, I = index.search(query_vector, k=n_docs)  # Busca los 5 documentos más similares
    retrieved_info = get_retrieved_info(documents, I, D)
    formatted_info = format_retrieved_info(retrieved_info)
    prompt = generate_prompt(query_text, formatted_info)
    answer = answer_using_ollama(prompt, model=llm_model)
    return answer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Medical Chatbot using RAG')
    parser.add_argument('--mode', type=int, help='Execution mode: 0 for console mode, 1 for GUI mode, 2 for generate test metrics.', default=0)
    parser.add_argument('--llm_model', type=str, help='LLM model', default='gemma:2b')
    parser.add_argument('--embedding_model', type=str, help='Embedding model.', default='TimKond/S-PubMedBert-MedQuAD')
    parser.add_argument('--n_samples', type=int, help='Number of relevant docs obtained in the retrieval.', default=5)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
    parser.add_argument('--device', type=str, help='Device', default='cuda')
    args = parser.parse_args()

    # Validar los argumentos
    assert args.mode in [0, 1, 2], "Enter a correct option."
    assert args.llm_model in ['llama2', 'gemma:2b', 'gemma:7b', 'medllama2', 'meditron:7b'], "Enter a correct LLM model."
    assert args.embedding_model in ['emilyalsentzer/Bio_ClinicalBERT', 'TimKond/S-PubMedBert-MedQuAD'], "Enter a correct embedding model."
    assert args.device in ['cuda', 'cpu'], "Enter a correct device."

    class CFG:
        mode = args.mode
        embedding_model = args.embedding_model
        batch_size = args.batch_size
        device = args.device
        llm = args.llm_model
        n_samples = args.n_samples

    # Show config
    config = CFG()
    config_items = {k: v for k, v in vars(CFG).items() if not k.startswith('__')}
    print(tabulate(config_items.items(), headers=['Parameter', 'Value'], tablefmt='fancy_grid'))

    # Obtener los datos y cargar o generar el índice
    df = get_all_data()
    documents = TextDataset(df)
    if not os.path.exists('./storage/faiss_index.faiss'):
        embeddings = get_bert_embeddings(documents, CFG.batch_size, CFG.embedding_model, CFG.device)
        index = create_faiss_index(embeddings)
        write_index(index, './storage/faiss_index.faiss')
    else:
        index = faiss.read_index('./storage/faiss_index.faiss')


    if args.mode == 0:# Modo consola
        while True:
            query_text = input("Enter your question (/exit to quit): ")
            if query_text == '/exit':
                break
            else:
                answer = answer_query(query_text, index, documents, CFG.llm, CFG.embedding_model, CFG.n_samples, CFG.device)
                print(f"Answer: {answer}\n\n")


    elif args.mode == 1:# Modo GUI
        def make_inference(query, hist):
            return answer_query(query, index, documents, CFG.llm, CFG.embedding_model, CFG.n_samples, CFG.device)
        
        demo = gr.ChatInterface(fn = make_inference, 
                        examples = ["What is diabetes?", "Is ginseng good for diabetes?", "What are the symptoms of diabetes?", "What is Celiac disease?"], 
                        title = "Medical RAG Chatbot", 
                        description = "Medical RAG Chatbot is a chatbot that can help you with your medical queries. It is not a replacement for a doctor. Please consult a doctor for any medical advice.",
                        )
        demo.launch()

    elif args.mode == 2:# Calcular métricas de evaluación
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize
        import nltk
        nltk.download('punkt')

        questions, reference_answers = load_test_dataset()
        generated_answers = [answer_query(query, index, documents, CFG.llm, CFG.embedding_model, CFG.n_samples, CFG.device) for i, query in tqdm(enumerate(questions)) if i < 50]

        # Calcular ROUGE
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, gen) for ref, gen in zip(reference_answers, generated_answers)]
        bleu_scores = [sentence_bleu([word_tokenize(ref)], word_tokenize(gen)) for ref, gen in zip(reference_answers, generated_answers)]

        # Calcular el promedio de ROUGE y BLEU scores
        avg_rouge1_f1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
        avg_rougeL_f1 = np.mean([score['rougeL'].fmeasure for score in rouge_scores])
        avg_bleu = np.mean(bleu_scores)

        print(f"Average ROUGE-1 F1 score: {avg_rouge1_f1}")
        print(f"Average ROUGE-L F1 score: {avg_rougeL_f1}")
        print(f"Average BLEU score: {avg_bleu}")





