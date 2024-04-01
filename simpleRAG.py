import os
from faiss import write_index
import gradio as gr
import numpy as np
import torch 
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import pandas as pd
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
from transformers import TextIteratorStreamer
from threading import Thread

torch.set_num_threads(2)


# OBTENER EL DATASET________________________________________________________________________________
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


def get_medquad_dataset(with_na=False):
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
    df_1 = get_medquad_dataset()
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


def answer_using_gemma(prompt, model, tokenizer):  
    model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    count_tokens = lambda text: len(tokenizer.tokenize(text))
    
    streamer = TextIteratorStreamer(tokenizer, timeout=540., skip_prompt=True, skip_special_tokens=True)
    
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=6000 - count_tokens(prompt),
        top_p=0.2,
        top_k=20,
        temperature=0.1,
        repetition_penalty=2.0,
        length_penalty=-0.5,
        num_beams=1
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()  # Starting the generation in a separate thread.
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
    return partial_message


def answer_query(query_text, index, documents, llm_model, llm_tokenizer, embedding_model, n_docs, device):
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
    answer = answer_using_gemma(prompt, llm_model, llm_tokenizer)
    return answer




if __name__ == '__main__':

    class CFG:
        embedding_model = 'TimKond/S-PubMedBert-MedQuAD'
        batch_size = 128
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        llm = 'google/gemma-2b-it'
        n_samples = 3

    # Show config
    config = CFG()
    # config_items = {k: v for k, v in vars(CFG).items() if not k.startswith('__')}
    # print(tabulate(config_items.items(), headers=['Parameter', 'Value'], tablefmt='fancy_grid'))

    
    # Obtener los datos y cargar o generar el índice
    df = get_all_data()
    documents = TextDataset(df)
    if not os.path.exists('./storage/faiss_index.faiss'):
        embeddings = get_bert_embeddings(documents, CFG.batch_size, CFG.embedding_model, CFG.device)
        index = create_faiss_index(embeddings)
        write_index(index, './storage/faiss_index.faiss')
    else:
        index = faiss.read_index('./storage/faiss_index.faiss')

    # Load the model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", quantization_config=quantization_config, torch_dtype=torch.float16, low_cpu_mem_usage=True)


    def make_inference(query, hist):
        return answer_query(query, index, documents, model, tokenizer, CFG.embedding_model, CFG.n_samples, CFG.device)
    
    demo = gr.ChatInterface(fn = make_inference, 
                    examples = ["What is diabetes?", "Is ginseng good for diabetes?", "What are the symptoms of diabetes?", "What is Celiac disease?"], 
                    title = "Gemma 2b MedicalQA Chatbot", 
                    description = "Gemma 2b Medical Chatbot is a chatbot that can help you with your medical queries. It is not a replacement for a doctor. Please consult a doctor for any medical advice.",
                    )
    demo.launch()






