import torch
import esm
from transformers import AlbertModel, AlbertTokenizer
from tqdm import tqdm
import math
import re


def get_protein_features(protein_list):
    model, alphabets = esm.pretrained.esm1_t6_43M_UR50S()
    batch_conveter = alphabets.get_batch_converter()
    feature_list = []
    n_batch = 2
    n_step = math.ceil(len(protein_list) / n_batch)
    print('Getting Protein Features.....')
    for i in tqdm(range(n_step)):
        if(i==n_step):
            buf_list = protein_list[i*n_batch:]
        else:
            buf_list = protein_list[i*n_batch:(i+1)*n_batch]
        batch_seq_list = []
        for j in range(len(buf_list)):
            batch_seq_list.append(('protein{}'.format(j+1), buf_list[j]))
        batch_label, batch_stars, batch_tokens = batch_conveter(batch_seq_list)
    
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6])
        token_embedding = results['representations'][6]
    
        for j, (_,seq) in enumerate(batch_seq_list):
            feature_list.append(token_embedding[j, 1:len(seq)+1].mean(0).numpy())
    print('Finished getting protein features')
    return feature_list

def albert_protein_features(protein_list, max_len=1000):
    model = AlbertModel.from_pretrained("Rostlab/prot_albert")
    tokenizer = AlbertTokenizer.from_pretrained("Rostlab/prot_albert", do_lower_case=False)
    model.eval()
    protein_embedding = []
    for protein in protein_list:
        protein += ' '*(max_len-len(protein))
        protein = [re.sub(r"[UZOBX]", "<unk>", p) for p in protein]
        ids = tokenizer.batch_encode_plus(protein, add_special_token=False, pad_to_max_length=False)
        input_ids = torch.tensor(ids['input_ids'])
        attention_mask = torch.tensor(ids['attention_mask'])
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        embedding = embedding.squeeze().numpy()
        protein_embedding.append(embedding)
    
    return protein_embedding
        