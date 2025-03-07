"""
Módulo para el manejo de datasets y carga de datos.
"""
import os
import pandas as pd
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import List, Dict, Any, Tuple, Optional


class TextDataset(Dataset):
    """
    Dataset personalizado para datos de texto médico.
    
    Attributes:
        questions (List[str]): Lista de preguntas
        answers (List[str]): Lista de respuestas correspondientes
        urls (List[str]): URLs de las fuentes de información
    """
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa el dataset con un dataframe de pandas.
        
        Args:
            df (pd.DataFrame): DataFrame con columnas 'question', 'answer' y 'url'
        """
        self.questions = df['question'].tolist()
        self.answers = df['answer'].tolist()
        self.urls = df['url'].tolist()
        
    def __len__(self) -> int:
        """Retorna el número de elementos en el dataset."""
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        Retorna un elemento del dataset por su índice.
        
        Args:
            idx (int): Índice del elemento a recuperar
            
        Returns:
            Dict[str, str]: Diccionario con la pregunta, respuesta y URL
        """
        return {
            'question': self.questions[idx],
            'answer': self.answers[idx],
            'url': self.urls[idx]
        }


def get_medical_flashcards() -> pd.DataFrame:
    """
    Carga el dataset de tarjetas médicas desde Hugging Face.
    
    Returns:
        pd.DataFrame: DataFrame con preguntas y respuestas médicas
    """
    dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")
    df = pd.DataFrame(dataset['train'], columns=['input', 'output'])
    df = df.drop_duplicates(subset=['output'])
    df = df.drop_duplicates(subset=['input'])
    df['url'] = 'Not provided.'
    df = df.rename(columns={'input': 'question', 'output': 'answer'})
    df = df[['question', 'answer', 'url']]
    return df


def get_medquad_dataset(processed_data_dir: str, with_na: bool = False) -> pd.DataFrame:
    """
    Carga los datos procesados del dataset MedQuAD.
    
    Args:
        processed_data_dir (str): Directorio con los archivos CSV procesados
        with_na (bool): Si se incluyen filas con valores NA
        
    Returns:
        pd.DataFrame: DataFrame combinado de todos los archivos
    """
    if not os.path.exists(processed_data_dir):
        print(f"Directorio no encontrado: {processed_data_dir}")
        return pd.DataFrame()
    
    files = os.listdir(processed_data_dir)
    dfs = []
    
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(processed_data_dir, file)
            try:
                df = pd.read_csv(file_path, na_values=['', ' ', 'No information found.'])
                dfs.append(df)
            except Exception as e:
                print(f"Error al cargar {file}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined_df = pd.concat(dfs, ignore_index=True)
    if not with_na:
        combined_df = combined_df.dropna()
    
    return combined_df


def get_all_data(processed_data_dir: str) -> pd.DataFrame:
    """
    Carga y combina todos los datasets disponibles.
    
    Args:
        processed_data_dir (str): Directorio con datos procesados
        
    Returns:
        pd.DataFrame: DataFrame combinado con todos los datos
    """
    medquad_df = get_medquad_dataset(processed_data_dir)
    flashcards_df = get_medical_flashcards()
    
    # Combinar datasets y asegurar el formato correcto
    combined_df = pd.concat([medquad_df, flashcards_df], ignore_index=True)
    combined_df = combined_df[['question', 'answer', 'url']]
    
    # Eliminar duplicados exactos
    combined_df = combined_df.drop_duplicates()
    
    print(f"Dataset cargado: {len(combined_df)} ejemplos")
    return combined_df


def load_test_dataset(test_file_path: str) -> Tuple[List[str], List[str]]:
    """
    Carga el dataset de prueba para evaluación.
    
    Args:
        test_file_path (str): Ruta al archivo CSV de prueba
        
    Returns:
        Tuple[List[str], List[str]]: Tupla de (preguntas, respuestas_referencia)
    """
    if not os.path.exists(test_file_path):
        print(f"Archivo de test no encontrado: {test_file_path}")
        return [], []
    
    try:
        df = pd.read_csv(test_file_path)
        pattern = r'Question:\s*(.*?)\s*URL:\s*(https?://[^\s]+)\s*Answer:\s*(.*)'
        questions_df = df['Answer'].str.extract(pattern, expand=True)
        questions_df.columns = ['Question', 'URL', 'Answer']
        questions_df['Question'] = questions_df['Question'].str.replace(r'\(Also called:.*?\)', '', regex=True).str.strip()
        
        questions = questions_df['Question'].tolist()
        answers_ground_truth = questions_df['Answer'].tolist()
        return questions, answers_ground_truth
    except Exception as e:
        print(f"Error al cargar el dataset de test: {e}")
        return [], []