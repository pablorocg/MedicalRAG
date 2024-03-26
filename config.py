class CFG:
    embedding_model = 'TimKond/S-PubMedBert-MedQuAD'
    n_samples = None
    with_na = False
    batch_size = 128
    device = 'cuda'
    log_embeddings = False
    llm = "medllama2"
    assert embedding_model in ['emilyalsentzer/Bio_ClinicalBERT', 'TimKond/S-PubMedBert-MedQuAD'], 'Invalid embedding model'
    assert llm in ['llama2', 'gemma:2b', 'gemma:7b', 'medllama2', 'meditron:7b'], 'Invalid LLM model'