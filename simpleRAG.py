#librerias necesarias
import os
import pandas as pd
import numpy as np
import faiss
from faiss import write_index
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import CFG
from datasets import load_dataset
import os
import requests
import json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def get_medical_flashcards_dataset():
    dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")
    df = pd.DataFrame(dataset['train'], columns=['input', 'output'])
    df = df.drop_duplicates(subset=['output'])
    df = df.drop_duplicates(subset=['input'])
    df['url'] = 'Not provided.'
    df = df.rename(columns={'input': 'question', 'output': 'answer'})
    df = df[['question', 'answer', 'url']]
    return df


def read_processed_data(with_na=False, n_samples=None):
    files = os.listdir('dataset/processed_data')
    for idx, file in enumerate(files):
        if idx == 0:
            df = pd.read_csv('dataset/processed_data/' + file, na_values=['', ' ', 'No information found.'])
        else:
            df = pd.concat([df, pd.read_csv('dataset/processed_data/' + file, na_values=['', ' ', 'No information found.'])], ignore_index=True)
    if not with_na:
        df = df.dropna()
    if n_samples is not None:
        df = df.sample(n_samples)
    return df

def get_all_data(with_na=False):
    df_1 = read_processed_data(with_na=with_na)
    df_2 = get_medical_flashcards_dataset()

    # Concatenate the two dataframes
    df = pd.concat([df_1, df_2], ignore_index=True)

    # Conservar solo las columnas question, answer y url
    df = df[['question', 'answer', 'url']]

    return df

class TextDataset(Dataset):
    def __init__(self, df):# Input is a pandas dataframe
        self.questions = df.question.tolist()
        self.answers = df.answer.tolist()
        self.url = df.url.tolist()
      
        
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {'Q': self.questions[idx], # Texto
                'A': self.answers[idx],
                'U': self.url[idx]}
    

def collate_fn(batch, tokenizer=AutoTokenizer.from_pretrained(CFG.embedding_model)):
    # Extrae las preguntas de los elementos del batch
    questions = [item['Q'] for item in batch] # Lista de textos 
    
    # Tokeniza las preguntas en un lote
    tokenized_questions = tokenizer(
        questions,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # No hay necesidad de usar pad_sequence aquí, ya que tokenizer maneja el padding
    return {
        "input_ids": tokenized_questions['input_ids'],
        "attention_mask": tokenized_questions['attention_mask']
    }

def get_bert_embeddings(ds, batch_size=CFG.batch_size):
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    model = AutoModel.from_pretrained(CFG.embedding_model)
    model = model.to(CFG.device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(CFG.device)
            attention_mask = batch['attention_mask'].to(CFG.device)
            outputs = model(input_ids, attention_mask)
            last_hidden_state = outputs.last_hidden_state
            cls_embedding = last_hidden_state[:, 0, :]
            embeddings.append(cls_embedding.cpu().numpy())
    return np.concatenate(embeddings)

# Función para crear el índice FAISS
def create_faiss_index(embeddings):
  dimension = embeddings.shape[1]
  index = faiss.IndexFlatL2(dimension)
  index.add(embeddings)
  return index


# Función para obtener los embeddings de una consulta de texto
def get_query_embedding(query_text, device = CFG.device):
    tokenizer = AutoTokenizer.from_pretrained(CFG.embedding_model)
    model = AutoModel.from_pretrained(CFG.embedding_model).to(device)
    inputs = tokenizer(query_text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(1).squeeze().cpu().numpy()
    return query_embedding

def get_retrieved_info(documents, I, D):
    retrieved_info = dict()
    for i, idx in enumerate(I[0], start=1):
        retrieved_info[i] = {
            "url": documents[idx]['U'],
            "question": documents[idx]['Q'],
            "answer": documents[idx]['A'],
        }
    return retrieved_info


def format_retrieved_info(retrieved_info):
    formatted_info = "\n"
    for i, info in retrieved_info.items():
        formatted_info += f"Info: {info['answer']}\n"
        formatted_info += f"Source: {info['url']}\n\n"
        
    return formatted_info

def generate_prompt(query_text, formatted_info):
    prompt = """
    As a specialized medical LLM, you're designed to provide informative, well-reasoned responses to health queries strictly based on the context provided, without relying on prior knowledge. Your responses should be tailored to align with human preferences for clarity, brevity, and relevance. 

    Question: "{query_text}"

    Considering only the context information:
    {formatted_info}
    
    Use the provided information to support your answer, ensuring it is clear, concise, and directly addresses the user's query. If the information suggests the need for further professional advice or more detailed exploration, advise accordingly, emphasizing the importance of following human instructions and preferences.
    """
    prompt = prompt.format(query_text=query_text, formatted_info=formatted_info)
    return prompt

def answer_using_ollama(prompt, streaming=False):
    
    full_response = []
    url = 'http://localhost:11434/api/generate'
    data = {
        "model": "gemma:2b", #Using llama2 7B params Q4 "gemma:2b" 
        "prompt": prompt
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)

    try:
        for line in response.iter_lines():
            if line:
                decoded_line = json.loads(line.decode('utf-8'))
                if streaming:
                    print(decoded_line['response'], end="")  # uncomment to results, token by token
                full_response.append(decoded_line['response'])
    finally:
        response.close()

    # response as string 
    return "".join(full_response)




def answer_query(query_text, index, documents):
    query_embedding = get_query_embedding(query_text)
    query_vector = np.expand_dims(query_embedding, axis=0)
    D, I = index.search(query_vector, k=5)  # Busca los 5 documentos más similares
    retrieved_info = get_retrieved_info(documents, I, D)
    formatted_info = format_retrieved_info(retrieved_info)
    prompt = generate_prompt(query_text, formatted_info)
    answer = answer_using_ollama(prompt)
    
    return answer

def load_test_dataset():
    
    df = pd.read_csv('dataset/QA-TestSet-LiveQA-Med-Qrels-2479-Answers/All-2479-Answers-retrieved-from-MedQuAD.csv')
    pattern = r'Question:\s*(.*?)\s*URL:\s*(https?://[^\s]+)\s*Answer:\s*(.*)'
    questions_df = df['Answer'].str.extract(pattern, expand=True)
    questions_df.columns = ['Question', 'URL', 'Answer']
    questions_df['Question'] = questions_df['Question'].str.replace(r'\(Also called:.*?\)', '', regex=True).str.strip()
    
    questions = questions_df['Question'].tolist()
    answers_groud_truth = questions_df['Answer'].tolist()
    return questions, answers_groud_truth


if __name__ == '__main__':

    # Cargar los datos
    df = get_all_data()
    documents = TextDataset(df)

    # si no existe el fichero faiss.index, se crea
    if not os.path.exists('faiss_index.faiss'):
        embeddings = get_bert_embeddings(documents, CFG.batch_size)
        index = create_faiss_index(embeddings)
        write_index(index, 'faiss_index.faiss')
    else:
        index = faiss.read_index('faiss_index.faiss')

    
    query = "What is the function of the nucleus?"
    answer = answer_query(query, index, documents)
    print(answer)

    query2 = "What is the function of the mitochondria?"
    answer2 = answer_query(query2, index, documents)
    print(answer2)
   

    # Evaluate QA responses using the test dataset
    questions, reference_answers = load_test_dataset()
    generated_answers = [answer_query(query, index, documents) for query in questions]

    # Calcular ROUGE y BLEU scores
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.tokenize import word_tokenize


    # Calcular ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, gen) for ref, gen in zip(reference_answers, generated_answers)]

    for scores in rouge_scores:
        print(scores)

    # Calcular BLEU
    bleu_scores = [sentence_bleu([word_tokenize(ref)], word_tokenize(gen)) for ref, gen in zip(reference_answers, generated_answers)]

    for score in bleu_scores:
        print(f"BLEU score: {score}")



