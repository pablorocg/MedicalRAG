class CFG:
    embedding_model = 'TimKond/S-PubMedBert-MedQuAD'
    n_samples = None
    with_na = True
    batch_size = 128
    device = 'cuda'
    log_embeddings = False

    assert embedding_model in ['emilyalsentzer/Bio_ClinicalBERT', 'TimKond/S-PubMedBert-MedQuAD'], 'Invalid embedding model'