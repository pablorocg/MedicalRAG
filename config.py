class CFG:
    embedding_model = 'TimKond/S-PubMedBert-MedQuAD'
    n_samples = None
    with_na = False
    batch_size = 128
    device = 'cuda'

    assert embedding_model in ['emilyalsentzer/Bio_ClinicalBERT', 'TimKond/S-PubMedBert-MedQuAD'], 'Invalid embedding model'