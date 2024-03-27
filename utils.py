from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset
import faiss
import os
import requests
import json
import os

import gradio as gr
import numpy as np
import pandas as pd
import requests



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

