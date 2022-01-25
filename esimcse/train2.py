import copy
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from loading import DataLoading
from config import Params
from model import ESimCSE
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def compute_loss(query, key, queue, tao=0.05):
    '''
    @function: 计算对比损失函数
    
    @input:
    query: tensor,查询原句向量
    key:   tensor,增强原句向量 
    queue: tensor,历史队列句向量
    tao:   float,温度系数，超参数，默认0.05
    
    @return: loss(tensor),损失函数值
    '''
    # N: batch, D: dim
    N, D = query.shape[0], query.shape[1]
    
    # calculate positive similarity
    pos = torch.exp(torch.div(torch.bmm(query.view(N,1,D), key.view(N,D,1)).view(N,1),tao))
    
    # calculate inner_batch similarity
    batch_all = torch.sum(torch.exp(torch.div(torch.mm(query.view(N,D),torch.t(key)),tao)),dim=1)
    # calculate inner_queue similarity
    queue_all = torch.sum(torch.exp(torch.div(torch.mm(query.view(N,D),torch.t(queue)),tao)),dim=1)
    
    denominator = batch_all + queue_all
    
    loss = torch.mean(-torch.log(torch.div(pos, denominator)))
    return loss


def compute_loss2(query, query2, key, queue, tao=0.05):
    '''
    @function: 计算对比损失函数
    
    @input:
    query: tensor,查询原句向量
    query2: tensor,经过dropout增强的查询原句向量
    key:   tensor,增强原句向量 
    queue: tensor,历史队列句向量
    tao:   float,温度系数，超参数，默认0.05
    
    @return: loss(tensor),损失函数值
    '''
    # N: batch, D: dim
    N, D = query.shape[0], query.shape[1]
    
    # calculate dropout augmentation positive similarity (numerator)
    dropout_pos = torch.exp(torch.div(torch.bmm(query.view(N,1,D), query2.view(N,D,1)).view(N,1),tao))
    # calculate dropout augmentation positive and negative similarity (denominator)
    dropout_all = torch.sum(torch.exp(torch.div(torch.mm(query.view(N,D),torch.t(query2)),tao)),dim=1)
    dropout_loss = torch.mean(-torch.log(torch.div(dropout_pos, dropout_all)))
    
    # calculate data augmentation positive similarity (numerator)
    pos = torch.exp(torch.div(torch.bmm(query.view(N,1,D), key.view(N,D,1)).view(N,1),tao))
    # calculate inner_batch similarity
    batch_all = torch.sum(torch.exp(torch.div(torch.mm(query.view(N,D),torch.t(key)),tao)),dim=1)
    # calculate inner_queue similarity
    queue_all = torch.sum(torch.exp(torch.div(torch.mm(query.view(N,D),torch.t(queue)),tao)),dim=1)
    
    denominator = batch_all + queue_all
    
    batch_data_loss = torch.mean(-torch.log(torch.div(pos, denominator)))
    
    loss = dropout_loss + batch_data_loss
    # print("dropout loss: {}, batch loss: {}".format(dropout_loss, batch_data_loss), flush=True)
    return loss
    

def construct_queue(train_question_dl, tokenizer, key_encoder, device):
    flag = 0
    queue = None
    while True:
        with torch.no_grad():
            for pid, pair in enumerate(train_question_dl):
                # 和初始数据不同的数据作为反例
                if pid < 1000:
                    continue
                key_question = list(pair[1])
                key_encodings = tokenizer(key_question,
                                          padding=True,
                                          truncation=True,
                                          max_length=Params.max_length,
                                          return_tensors='pt')
                key_embedding = key_encoder(key_encodings['input_ids'].to(device),
                                            key_encodings['attention_mask'].to(device),
                                            key_encodings['token_type_ids'].to(device))
                if queue is None:
                    queue = key_embedding
                else:
                    if queue.shape[0] < Params.queue_num:
                        queue = torch.cat((queue, key_embedding), 0)
                    else:
                        flag = 1
                if flag == 1:
                    break
        if flag == 1:
            break
    queue = queue[:Params.queue_num]
    queue = torch.div(queue, torch.norm(queue, dim=1).reshape(-1, 1))
    return queue
    

def train():
    data_loading = DataLoading(Params.question_file, Params.qa_file, Params.pretrained_model)
    questions = data_loading.load_data()
    print("load dataset done, data num: {}!".format(len(questions)))
    pos_question_pairs = data_loading.generate_pos_dataset(questions)
    print("data augmentation done!")
    train_question_dl = DataLoader(pos_question_pairs,
                                   batch_size=Params.batch_size,
                                   shuffle=True)
    batches_num = len(train_question_dl)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_encoder = ESimCSE(Params.pretrained_model, Params.dropout)
    key_encoder = copy.deepcopy(query_encoder)
    query_encoder = query_encoder.to(device)
    query_encoder.train()
    key_encoder = key_encoder.to(device)
    optimizer = torch.optim.AdamW(query_encoder.parameters(), lr=Params.lr)
    
    best_loss = 100000
    queue_embeddings = construct_queue(train_question_dl, data_loading.tokenizer, key_encoder, device)
    print("create negative queue done!", flush=True)
    print("start training...", flush=True)
    for epoch in range(Params.epoches):
        epoch_loss = []
        batch = 0
        for pairs in tqdm(train_question_dl):
            query, key = list(pairs[0]), list(pairs[1])
            query_encodings = data_loading.tokenizer(query,
                                                     padding=True,
                                                     truncation=True,
                                                     max_length=Params.max_length,
                                                     return_tensors='pt')
            query_embeddings = query_encoder(query_encodings['input_ids'].to(device),
                                             query_encodings['attention_mask'].to(device),
                                             query_encodings['token_type_ids'].to(device))
            query_embeddings2 = query_encoder(query_encodings['input_ids'].to(device),
                                             query_encodings['attention_mask'].to(device),
                                             query_encodings['token_type_ids'].to(device))
            
            key_encodings = data_loading.tokenizer(key,
                                                   padding=True,
                                                   truncation=True,
                                                   max_length=Params.max_length,
                                                   return_tensors='pt')
            key_embeddings = key_encoder(key_encodings['input_ids'].to(device),
                                         key_encodings['attention_mask'].to(device),
                                         key_encodings['token_type_ids'].to(device)).detach()
            
            # normalize sentence embedding
            query_embeddings = torch.div(query_embeddings, torch.norm(query_embeddings, dim=1).reshape(-1, 1))
            query_embeddings2 = torch.div(query_embeddings2, torch.norm(query_embeddings2, dim=1).reshape(-1, 1))
            key_embeddings = torch.div(key_embeddings, torch.norm(key_embeddings, dim=1).reshape(-1, 1))
            
            # compute loss
            loss = compute_loss2(query_embeddings, query_embeddings2, key_embeddings, queue_embeddings, Params.tao)
            epoch_loss.append(loss.item())
            
            # update gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # update queue
            queue_embeddings = torch.cat((queue_embeddings, key_embeddings), 0)
            if queue_embeddings.shape[0] > Params.queue_num:
                queue_embeddings = queue_embeddings[Params.batch_size:, :]
            
            # update key encoder momentum 
            for query_params, key_params in zip(query_encoder.parameters(), key_encoder.parameters()):
                key_params.data.copy_(Params.momentum * key_params + (1-Params.momentum) * query_params)
                key_params.requires_grad = False
             
            if batch % Params.display_interval == 0:
                print("Epoch: {}, batch: {}/{}, loss: {}".format(epoch, batch, batches_num, loss.item()), flush=True)
            batch += 1
        
        avg_epoch_loss = np.mean(epoch_loss)
        print("Epoch: {}, avg loss: {}".format(epoch, avg_epoch_loss), flush=True)
        if avg_epoch_loss < best_loss:
            print("Epoch: {}, loss: {}, save best model".format(epoch, avg_epoch_loss), flush=True)
            model_path = Params.esimcse_model + "_{}_{}.pth".format(Params.dropout, Params.queue_num)
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch,
                'loss': avg_epoch_loss,
                'model_state_dict': query_encoder.state_dict()
            }, model_path)

    
train()

# query= torch.randn([3, 4])
# key = torch.randn([3, 4])

# query= torch.div(query, torch.norm(query, dim=1).reshape(-1,1))
# key = torch.div(key, torch.norm(key, dim=1).reshape(-1,1))

# queue = torch.randn([10, 4])
# loss1 = compute_loss(query, key, queue)

# queue = torch.div(queue, torch.norm(queue, dim=1).reshape(-1,1))
# loss2 = compute_loss(query, key, queue)
# print(loss1, loss2)
