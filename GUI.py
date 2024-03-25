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
import gradio as gr

import requests
import json
from scipy.spatial.distance import cosine

from llama_index.core import (
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    SummaryIndex,
    Document
)

from llama_index.vector_stores.faiss import FaissVectorStore

class MedicalChatBotGUI():
    def __init__(self,model):
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        self.model=model
        self.vector_store = None
        self.index = None
        self.storage_context = None
        self.data_generator = None
        self.index = None
        self.documents = None

    def read_processed_data(self,with_na=False, n_samples=None):
        # List the files in the processed_data directory
        files = os.listdir('dataset/processed_data')

        # Read the files into a dataframe
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
    def collate_fn(self,batch, tokenizer=AutoTokenizer.from_pretrained(CFG.embedding_model)):
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


    def get_bert_embeddings(self,ds, batch_size=CFG.batch_size):
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn, drop_last=False)
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
    def create_faiss_index(self,embeddings):
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index


    # Función para obtener los embeddings de una consulta de texto
    def get_query_embedding(self,query_text, device = CFG.device):
        tokenizer = AutoTokenizer.from_pretrained(CFG.embedding_model)
        model = AutoModel.from_pretrained(CFG.embedding_model).to(device)
        inputs = tokenizer(query_text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(1).squeeze().cpu().numpy()
        return query_embedding


    def get_retrieved_info(self,documents, I, D):
        retrieved_info = dict()
        for i, idx in enumerate(I[0], start=1):
            retrieved_info[i] = {
                "url": documents[idx]['U'],
                "question": documents[idx]['Q'],
                "answer": documents[idx]['A'],
                "dissimilarity": D[0][i-1]
            }
        return retrieved_info


    def format_retrieved_info(self,retrieved_info):
        formatted_info = "\n"
        for i, info in retrieved_info.items():
            formatted_info += f"Question: {info['question']}\n"
            formatted_info += f"Answer: {info['answer']}\n"
            formatted_info += f"Source: {info['url']}\n\n"
        return formatted_info


    def generate_prompt(self,query_text, formatted_info):
        prompt = """ 
        You are a medical sciences bot tailored for precision and succinctness. 
        Your programming dictates responding directly to the user's query with utmost brevity. 
        Your key task is to evaluate the user's question against your vast database of documents. 
        The lower the dissimilarity between the query and the document, the more emphasis you should place on that information in your response. 
        Your recommendation should be concise, backed by a URL to the most pertinent document for user reference, serving as proof of the recommendation's validity. 
        Swift and relevant information retrieval is your principal function.

        Given the user's question: {query_text}

        And taking into account the pertinent information: 
        {formatted_info}

        Formulate a targeted recommendation for the user. 
        The recommendation should be aligned closely with their query.  
        """
        prompt = prompt.format(query_text=query_text, formatted_info=formatted_info)

        return prompt


    def answer_using_ollama(self,prompt,model):
        
        full_response = []
        url = 'http://localhost:11434/api/generate'
        data = {
            "model": model, #Using llama2 7B params Q4 "gemma:2b" "gemma:7b"
            "prompt": prompt
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)

        try:
            for line in response.iter_lines():
                if line:
                    decoded_line = json.loads(line.decode('utf-8'))
                    print(decoded_line['response'], end="")  # uncomment to results, token by token
                    full_response.append(decoded_line['response'])
        finally:
            response.close()

        # response as string 
        return "".join(full_response)
    def generate_faiss_db(self):

        df = self.read_processed_data(with_na = CFG.with_na, n_samples=CFG.n_samples)
        self.documents = TextDataset(df)
        embeddings = self.get_bert_embeddings(self.documents, CFG.batch_size)
        self.index = self.create_faiss_index(embeddings)# Crea el índice FAISS con los embeddings
        # list_of_docs = df.question.tolist()
        # documents = [Document(text = t) for t in list_of_docs]
        # self.vector_store = FaissVectorStore(faiss_index=index)
        # self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store, embed_model='local')
        # llamaindex = VectorStoreIndex.from_documents(
        #     documents, storage_context=self.storage_context
        # )
        # llamaindex.storage_context.persist()
        
        write_index(self.index, 'faiss_index.faiss')# Guarda el índice FAISS en un archivo binario
        return df

    def get_source_answer(self, answer, retrieved_info):
        # Obten el embedding de la respuesta
        answer_embedding = self.get_query_embedding(answer)
        
        # Inicializa la menor distancia encontrada y la URL del documento más cercano
        min_distance = float('inf')
        closest_url = ""
        
        # Itera a través de la información recuperada para encontrar el documento más cercano
        for info in retrieved_info.values():
            document_embedding = self.get_query_embedding(info['answer']) # Podrías necesitar ajustar esto si ya tienes los embeddings
            distance = cosine(answer_embedding, document_embedding)
            
            if distance < min_distance:
                min_distance = distance
                closest_url = info['url']
        
        return closest_url
    def make_inference(self,query, hist=None):
        df = self.read_processed_data(with_na = CFG.with_na, n_samples=CFG.n_samples)
        documents = TextDataset(df)
        # embeddings = get_bert_embeddings(documents, CFG.batch_size)
        # index = create_faiss_index(embeddings)# Crea el índice FAISS con los embeddings
        index = faiss.read_index("faiss_index.faiss")
        query_embedding = self.get_query_embedding(query)
        query_vector = np.expand_dims(query_embedding, axis=0)
        D, I = index.search(query_vector, k=5)  # Busca los 5 documentos más similares
        print(D, I)
        retrieved_info = self.get_retrieved_info(documents, I, D)
        print(retrieved_info)
        formatted_info = self.format_retrieved_info(retrieved_info)
        print(formatted_info)
        prompt = self.generate_prompt(query, formatted_info)
        answer = self.answer_using_ollama(prompt,self.model)
        source = self.get_source_answer(answer,retrieved_info)
        return answer + ' \n Source: '+ source 


    def get_index_from_local_storage(self):
        self.vector_store = FaissVectorStore.from_persist_dir("faiss_index")
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store, persist_dir="faiss_index"
        )
        self.index = load_index_from_storage(storage_context=self.storage_context)



# CArga de datos y generacion de la BBDD vectorial
        
# query_text = "What is the cause of diabetes?"
# query_embedding = get_query_embedding(query_text)
# query_vector = np.expand_dims(query_embedding, axis=0)
# D, I = index.search(query_vector, k=5)  # Busca los 5 documentos más similares
# retrieved_info = get_retrieved_info(documents, I, D)
# formatted_info = format_retrieved_info(retrieved_info)
# prompt = generate_prompt(query_text, formatted_info)
# answer = answer_using_ollama(prompt)

class TextDataset(Dataset):
    def __init__(self, df):# Input is a pandas dataframe
        self.questions = df.question.tolist()
        self.question_ids = df.question_id.tolist()
        self.question_types = df.question_type.tolist()
        self.answers = df.answer.tolist()
        self.focus = df.focus.tolist()
        self.doc_id = df.id.tolist()
        self.source = df.source.tolist()
        self.url = df.url.tolist()
        self.cui = df.cui.tolist()
        self.semantic_type = df.semanticType.tolist()
        self.semantic_group = df.semanticGroup.tolist()
        
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {'Q': self.questions[idx], # Texto
                'Q_id': self.question_ids[idx], 
                'Q_T': self.question_types[idx], 
                'A': self.answers[idx],
                'F': self.focus[idx],
                'D_id': self.doc_id[idx],
                'S': self.source[idx],
                'U': self.url[idx],
                'C': self.cui[idx],
                'S_T': self.semantic_type[idx],
                'S_G': self.semantic_group[idx]}