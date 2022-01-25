import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from config import Params
from model import PromptBERT
from tqdm import tqdm
from datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        
def read_sentence(corpus_file, tokenizer, prompt_templates):
    sentences = []
    with open(corpus_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            words_num = len(tokenizer.tokenize(line))
            sentence_pair = []
            for template in prompt_templates:
                # template_num = len(tokenizer.tokenize(template))
                if words_num > Params.max_length - 15:
                    line = line[:Params.max_length - 15]
                line_num = len(tokenizer.tokenize(line))
                prompt_line = template.replace(Params.replace_token, line)
                template_line = template.replace(Params.replace_token, Params.replace_token * line_num)
                sentence_pair.append([prompt_line, template_line])
            sentences.append(sentence_pair)
    return sentences


def compute_loss(query, key, tao=0.05):
    query = torch.div(query, torch.norm(query, dim=1).reshape(-1, 1))
    key = torch.div(key, torch.norm(key, dim=1).reshape(-1, 1))
    # print(query.shape, key.shape)
    N, D = query.shape[0], query.shape[1]
    
    # calculate positive similarity
    batch_pos = torch.exp(torch.div(torch.bmm(query.view(N, 1, D), key.view(N, D, 1)).view(N, 1), tao))
    
    # calculate inner_batch all similarity
    batch_all = torch.sum(torch.exp(torch.div(torch.mm(query.view(N, D), torch.t(key)), tao)), dim=1)
    
    loss = torch.mean(-torch.log(torch.div(batch_pos, batch_all)))
    return loss


def train():
    tokenizer = AutoTokenizer.from_pretrained(Params.pretrained_model_path)
    special_token_dict = {'additional_special_tokens': ['[X]']}
    tokenizer.add_special_tokens(special_token_dict)
    mask_id = tokenizer.convert_tokens_to_ids(Params.mask_token)
    sentences = read_sentence(Params.corpus_file, tokenizer, Params.prompt_templates)
    dl = DataLoader(sentences,
                    batch_size=Params.batch_size)
    model = PromptBERT(Params.pretrained_model_path, Params.dropout, mask_id)
    model.encoder.resize_token_embeddings(len(tokenizer))
    optimizer = optim.AdamW(model.parameters(), lr=Params.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model.to(device)
    
    batch_idx = 0
    min_loss = 100000
    
    for epoch in range(Params.epoches):
        epoch_loss = []
        for data in tqdm(dl):
            prompt_data0, prompt_data1 = data
            prompt_lines0, template_lines0 = prompt_data0
            prompt_lines1, template_lines1 = prompt_data1
            
            prompt_encodings0 = tokenizer(list(prompt_lines0),
                                         padding=True,
                                         truncation=True,
                                         max_length=Params.max_length,
                                         return_tensors='pt')
            prompt_encodings1 = tokenizer(list(prompt_lines1),
                                         padding=True,
                                         truncation=True,
                                         max_length=Params.max_length,
                                         return_tensors='pt')
            
            template_encodings0 = tokenizer(list(template_lines0),
                                           padding=True,
                                           truncation=True,
                                           max_length=Params.max_length,
                                           return_tensors='pt')
            template_encodings1 = tokenizer(list(template_lines1),
                                           padding=True,
                                           truncation=True,
                                           max_length=Params.max_length,
                                           return_tensors='pt')

            query = model(prompt_input_ids=prompt_encodings0['input_ids'].to(device),
                         prompt_attention_mask=prompt_encodings0['attention_mask'].to(device), 
                         prompt_token_type_ids=prompt_encodings0['token_type_ids'].to(device), 
                         template_input_ids=template_encodings0['input_ids'].to(device), 
                         template_attention_mask=template_encodings0['attention_mask'].to(device), 
                         template_token_type_ids=template_encodings0['token_type_ids'].to(device))
            key = model(prompt_input_ids=prompt_encodings1['input_ids'].to(device),
                         prompt_attention_mask=prompt_encodings1['attention_mask'].to(device), 
                         prompt_token_type_ids=prompt_encodings1['token_type_ids'].to(device), 
                         template_input_ids=template_encodings1['input_ids'].to(device), 
                         template_attention_mask=template_encodings1['attention_mask'].to(device), 
                         template_token_type_ids=template_encodings1['token_type_ids'].to(device))
            loss = compute_loss(query, key, Params.tao)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            epoch_loss.append(loss)
            if batch_idx % Params.display_interval == 0:
                print("Epoch: {}, batch: {}, loss: {}".format(epoch, batch_idx, loss), flush=True)
            batch_idx += 1
        avg_epoch_loss = np.mean(epoch_loss)
        print("Epoch: {}, batch: {}, avg loss: {}".format(epoch, batch_idx, avg_epoch_loss), flush=True)
        if avg_epoch_loss < min_loss:
            print("Epoch: {}, loss: {}, save best model".format(epoch, avg_epoch_loss), flush=True)
            min_loss = avg_epoch_loss
            localtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_epoch_loss,
                'datetime': localtime
            }, Params.promptbert_path)


if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained(Params.pretrained_model_path)
    # mask_id = tokenizer.convert_tokens_to_ids(Params.mask_token)
    # sentences = read_sentence(Params.corpus_file, tokenizer, Params.prompt_templates)
    # dl = DataLoader(sentences,
    #                 batch_size=Params.batch_size)
    # for data in dl:
    #     # print(data[0][0])
    #     print(data[0])
    #     print(data[1])
    #     exit(0)
    
    train()