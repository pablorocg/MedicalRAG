from transformers import AutoModel, AutoTokenizer
import torch 
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np


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