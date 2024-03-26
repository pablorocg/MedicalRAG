import json
import os

import faiss
from faiss import write_index
import gradio as gr
import numpy as np
import pandas as pd
import requests
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

# Set environment variable to avoid MKL errors
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def get_medical_flashcards_dataset():
    """
    Retrieves a medical flashcards dataset.

    Returns:
        df (pandas.DataFrame): A DataFrame containing the medical flashcards dataset.
            The DataFrame has three columns: 'question', 'answer', and 'url'.
    """
    dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")
    df = pd.DataFrame(dataset['train'], columns=['input', 'output'])
    df = df.drop_duplicates(subset=['output'])
    df = df.drop_duplicates(subset=['input'])
    df['url'] = 'Not provided.'
    df = df.rename(columns={'input': 'question', 'output': 'answer'})
    df = df[['question', 'answer', 'url']]
    return df


def read_processed_data(with_na=False):
    """
    Read and process data from multiple CSV files.

    Args:
        with_na (bool, optional): Whether to include rows with missing values. Defaults to False.
        n_samples (int, optional): Number of random samples to select from the data. Defaults to None.

    Returns:
        pandas.DataFrame: Processed data from the CSV files.
    """
    files = os.listdir('dataset/processed_data')
    for idx, file in enumerate(files):
        if idx == 0:
            df = pd.read_csv('dataset/processed_data/' + file, na_values=['', ' ', 'No information found.'])
        else:
            df = pd.concat([df, pd.read_csv('dataset/processed_data/' + file, na_values=['', ' ', 'No information found.'])], ignore_index=True)
    if not with_na:
        df = df.dropna()
    return df


def get_all_data():
    """
    Retrieves all data by combining processed data and medical flashcards dataset.

    Parameters:
        with_na (bool): Flag indicating whether to include records with missing values. Default is False.

    Returns:
        pandas.DataFrame: Combined dataframe with columns 'question', 'answer', and 'url'.
    """
    df_1 = read_processed_data()
    df_2 = get_medical_flashcards_dataset()
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = df[['question', 'answer', 'url']]
    return df

def load_test_dataset():
    """
    Load the test dataset from a CSV file and extract the questions and ground truth answers.

    Returns:
        questions (list): A list of questions extracted from the dataset.
        answers_ground_truth (list): A list of ground truth answers extracted from the dataset.
    """
    df = pd.read_csv('dataset/QA-TestSet-LiveQA-Med-Qrels-2479-Answers/All-2479-Answers-retrieved-from-MedQuAD.csv')
    pattern = r'Question:\s*(.*?)\s*URL:\s*(https?://[^\s]+)\s*Answer:\s*(.*)'
    questions_df = df['Answer'].str.extract(pattern, expand=True)
    questions_df.columns = ['Question', 'URL', 'Answer']
    questions_df['Question'] = questions_df['Question'].str.replace(r'\(Also called:.*?\)', '', regex=True).str.strip()
    
    questions = questions_df['Question'].tolist()
    answers_ground_truth = questions_df['Answer'].tolist()
    return questions, answers_ground_truth


class TextDataset(Dataset):
    """
    A custom dataset class for text data.

    Args:
        df (pandas.DataFrame): Input pandas dataframe containing the text data.

    Attributes:
        questions (list): List of questions from the dataframe.
        answers (list): List of answers from the dataframe.
        url (list): List of URLs from the dataframe.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the data at the given index.

    """

    def __init__(self, df):
        self.questions = df.question.tolist()
        self.answers = df.answer.tolist()
        self.url = df.url.tolist()

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {'Q': self.questions[idx],
                'A': self.answers[idx],
                'U': self.url[idx]}


def collate_fn(batch, embedding_model):
    """
    Collate function for processing a batch of data.

    Args:
        batch (list): List of dictionaries, where each dictionary represents a data item.
        tokenizer (Tokenizer): Tokenizer object used for tokenization (default: AutoTokenizer.from_pretrained(CFG.embedding_model)).

    Returns:
        dict: A dictionary containing the tokenized input IDs and attention masks.

    """
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    # Extract the questions from the batch items
    questions = [item['Q'] for item in batch]  # List of texts

    # Tokenize the questions in a batch
    tokenized_questions = tokenizer(
        questions,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512
    )

    # No need to use pad_sequence here, as tokenizer handles the padding
    return {
        "input_ids": tokenized_questions['input_ids'],
        "attention_mask": tokenized_questions['attention_mask']
    }


def get_bert_embeddings(ds, batch_size, embedding_model, device, collate_fn=collate_fn):
    """
    Get BERT embeddings for a given dataset.

    Args:
        ds (Dataset): The dataset containing input data.
        batch_size (int, optional): The batch size for data loading. Defaults to CFG.batch_size.

    Returns:
        numpy.ndarray: Concatenated BERT embeddings for all input data.
    """
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    model = AutoModel.from_pretrained(embedding_model)
    model = model.to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            last_hidden_state = outputs.last_hidden_state
            cls_embedding = last_hidden_state[:, 0, :]
            embeddings.append(cls_embedding.cpu().numpy())
    return np.concatenate(embeddings)


def create_faiss_index(embeddings):
    """
    Creates a Faiss index for the given embeddings.

    Parameters:
    embeddings (numpy.ndarray): The embeddings to be indexed.

    Returns:
    faiss.IndexFlatL2: The Faiss index object.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Función para obtener los embeddings de una consulta de texto
def get_query_embedding(query_text, device, embedding_model):
    """
    Get the embedding representation of a query text using a pre-trained model.

    Args:
        query_text (str): The input query text.
        device (str): The device to run the model on (default: CFG.device).

    Returns:
        numpy.ndarray: The query embedding as a numpy array.
    """
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model).to(device)
    inputs = tokenizer(query_text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(1).squeeze().cpu().numpy()
    return query_embedding

def get_retrieved_info(documents, I, D):
    """
    Retrieves information from a list of documents based on the given indices.

    Args:
        documents (list): A list of documents.
        I (tuple): A tuple containing the indices of the retrieved documents.
        D (dict): A dictionary containing the document information.

    Returns:
        dict: A dictionary containing the retrieved information, with the index as the key and the document information as the value.
    """
    retrieved_info = dict()
    for i, idx in enumerate(I[0], start=1):
        retrieved_info[i] = {
            "url": documents[idx]['U'],
            "question": documents[idx]['Q'],
            "answer": documents[idx]['A'],
        }
    return retrieved_info


def format_retrieved_info(retrieved_info):
    """
    Formats the retrieved information into a readable string.

    Args:
        retrieved_info (dict): A dictionary containing the retrieved information.

    Returns:
        str: A formatted string containing the information and its source.

    """
    formatted_info = "\n"
    for i, info in retrieved_info.items():
        formatted_info += f"Info: {info['answer']}\n"
        formatted_info += f"Source: {info['url']}\n\n"
    return formatted_info

def generate_prompt(query_text, formatted_info):
    """
    Generates a prompt for a specialized medical LLM to provide informative, well-reasoned responses to health queries.

    Parameters:
    query_text (str): The text of the health query.
    formatted_info (str): The formatted context information.

    Returns:
    str: The generated prompt.
    """
    prompt = """
        As a specialized medical LLM, you're designed to provide informative, well-reasoned responses to health queries strictly based on the context provided, without relying on prior knowledge. 
        Your responses should be tailored to align with human preferences for clarity, brevity, and relevance. 

        User question: "{query_text}"

        Considering only the context information:
        {formatted_info}
        
        Use the provided information to support your answer, ensuring it is clear, concise, and directly addresses the user's query. 
        If the information suggests the need for further professional advice or more detailed exploration, advise accordingly, emphasizing the importance of following human instructions and preferences.
    """
    prompt = prompt.format(query_text=query_text, formatted_info=formatted_info)
    return prompt

def answer_using_ollama(prompt, model):
    """
    Generates a response using the Ollama language model.

    Args:
        prompt (str): The prompt to generate a response for.
        streaming (bool, optional): Whether to stream the response token by token. Defaults to False.

    Returns:
        str: The generated response as a string.
    """
    full_response = []
    url = 'http://localhost:11434/api/generate'
    data = {
        "model": model, #Using llama2 7B params Q4 "gemma:2b" 
        "prompt": prompt
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)

    try:
        for line in response.iter_lines():
            if line:
                decoded_line = json.loads(line.decode('utf-8'))
                # if streaming:
                #     print(decoded_line['response'], end="")  # uncomment to results, token by token
                full_response.append(decoded_line['response'])
    finally:
        response.close()

    # response as string 
    return "".join(full_response)

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
    import argparse
    from tabulate import tabulate
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Medical Chatbot using RAG')
    parser.add_argument('--mode', type=int, help='Modo de ejecución: 0 para modo consola, 1 para modo GUI con gradio, 2 para calculo de metricas de test.', default=0)
    parser.add_argument('--llm_model', type=str, help='LLM model', default='gemma:2b')
    parser.add_argument('--embedding_model', type=str, help='Embedding model', default='TimKond/S-PubMedBert-MedQuAD')
    parser.add_argument('--n_samples', type=int, help='Número de docs obtenidos en retrieval', default=5)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
    parser.add_argument('--device', type=str, help='Device', default='cuda')
    args = parser.parse_args()

    # Validar los argumentos
    assert args.mode in [0, 1, 2], "Introduzca una opcion correcta"
    assert args.llm_model in ['llama2', 'gemma:2b', 'gemma:7b', 'medllama2', 'meditron:7b'], "Introduzca un modelo LLM correcto"
    assert args.embedding_model in ['emilyalsentzer/Bio_ClinicalBERT', 'TimKond/S-PubMedBert-MedQuAD'], "Introduzca un modelo de embedding correcto"
    assert args.device in ['cuda', 'cpu'], "Introduzca un dispositivo correcto"
    
    class CFG:
        mode = args.mode
        embedding_model = args.embedding_model
        batch_size = args.batch_size
        device = args.device
        llm = args.llm_model
        n_samples = args.n_samples


    config = CFG()
    config_items = {k: v for k, v in vars(CFG).items() if not k.startswith('__')}
    print(tabulate(config_items.items(), headers=['Param', 'Value'], tablefmt='fancy_grid'))

    # Obtener los datos y cargar o generar el índice
    df = get_all_data()
    documents = TextDataset(df)
    if not os.path.exists('faiss_index.faiss'):
        embeddings = get_bert_embeddings(documents, CFG.batch_size, CFG.embedding_model, CFG.device)
        index = create_faiss_index(embeddings)
        write_index(index, 'faiss_index.faiss')
    else:
        index = faiss.read_index('faiss_index.faiss')


    if args.mode == 0:# Modo consola
        while True:
            query_text = input("Introduzca su pregunta (exit para salir): ")
            if query_text == 'exit':
                break
            else:
                answer = answer_query(query_text, index, documents, CFG.llm, CFG.embedding_model, CFG.n_samples, CFG.device)
                print(f"Respuesta: {answer}\n\n")


    elif args.mode == 1:# Modo GUI
        def make_inference(query, hist):
            return answer_query(query, index, documents, CFG.llm, CFG.embedding_model, CFG.n_samples, CFG.device)
        
        demo = gr.ChatInterface(fn = make_inference, 
                        examples = ["What is diabetes?", "Is ginseng good for diabetes?", "What are the symptoms of diabetes?", "What are the symptoms of ABCD syndrome ?"], 
                        title = "Medical Chatbot", 
                        description = "Medical RAG Chatbot is a chatbot that can help you with your medical queries. It is a rule-based chatbot that can answer your queries based on the information it has. It is not a replacement for a doctor. Please consult a doctor for any medical advice.",
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





