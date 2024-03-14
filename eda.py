import pandas as pd
import os

# List files in dataset\processed_data

files = os.listdir('dataset/processed_data')
print(files)

# Read the files into a dataframe
for idx, file in enumerate(files):
    if idx == 0:
        df = pd.read_csv('dataset/processed_data/' + file, na_values=['', ' ', 'No information found.'])
    else:
        df = pd.concat([df, pd.read_csv('dataset/processed_data/' + file, na_values=['', ' ', 'No information found.'])], ignore_index=True)

# Quedarse con las filas sin nan
df = df.dropna()
# Quedarse con las 300 primeras filas
df = df.head(10000)

import faiss
from faiss import write_index
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Movemos el modelo y los datos al dispositivo que corresponda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Función para obtener word embeddings con BERT
def get_bert_embeddings(documents):
  tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
  model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
  embeddings = []
  for doc in documents:
    inputs = tokenizer(doc, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
      outputs = model(**inputs)
    doc_embedding = outputs.last_hidden_state.mean(1).squeeze().cpu().numpy()
    embeddings.append(doc_embedding)
  return np.array(embeddings)

# Función para crear el índice FAISS
def create_faiss_index(embeddings):
  dimension = embeddings.shape[1]
  index = faiss.IndexFlatL2(dimension)
  index.add(embeddings)
  return index


documents = df.answer.tolist()  # Los textos están en la columna "title"
print(documents)
# Procesamos los documentos para obtener los word embeddings
embeddings = get_bert_embeddings(documents)

# Crea el índice FAISS con los embeddings
index = create_faiss_index(embeddings)


# Función para obtener los embeddings de una consulta de texto
def get_query_embedding(query_text):
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
    inputs = tokenizer(query_text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(1).squeeze().cpu().numpy()
    return query_embedding

# Ejemplo de consulta
query_text = "Do I need to see a doctor for Abdominal mass ?"
query_embedding = get_query_embedding(query_text)
query_vector = np.expand_dims(query_embedding, axis=0)
print(f'Query vector: {query_vector}')
# Realiza la búsqueda en el índice FAISS
D, I = index.search(query_vector, k=5)  # Busca los 5 documentos más similares

print("Consulta realizada con el texto:", query_text)
print("Documentos más similares:")
for i, idx in enumerate(I[0], start=1):
    print(f"{i}: Documento {idx} con una distancia de {D[0][i-1]}")
    print("Contenido del documento:", documents[idx])