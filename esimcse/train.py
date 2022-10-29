# -*- coding: utf-8 -*-
# @Time    : 2022/9/20
# @Author  : Maciel


from transformers import AutoTokenizer
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import scipy.stats
from loading import MatchingDataSet
from ESimCSE import ESimCSE
import random
import argparse
from loguru import logger
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trainfile", type=str, default="../dataset/STS-B/train.txt", help="train file path")
    parser.add_argument("--devfile", type=str, default="../dataset/STS-B/dev.txt", help="dev file path")
    parser.add_argument("--testfile", type=str, default="../dataset/STS-B/test.txt", help="test file path")
    parser.add_argument("--filetype", type=str, default="txt", help="train and dev file type")
    parser.add_argument("--pretrained", type=str, default="hfl/chinese-roberta-wwm-ext", help="huggingface pretrained model")
    parser.add_argument("--model_out", type=str, default="../models/esimcse_roberta_stsb.pth", help="model output path")
    parser.add_argument("--dup_rate", type=float, default=0.2, help="repeat word probability")
    parser.add_argument("--queue_size", type=int, default=0.5, help="negative queue num / batch size")
    parser.add_argument("--momentum", type=float, default=0.995, help="momentum parameter")
    parser.add_argument("--max_length", type=int, default=128, help="sentence max length")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=10, help="epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--tao", type=float, default=0.05, help="temperature")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--display_interval", type=int, default=100, help="display interval")
    parser.add_argument("--pool_type", type=str, default="avg_first_last", help="pool_type")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="dropout_rate")
    parser.add_argument("--task", type=str, default="esimcse", help="task name")
    args = parser.parse_args()
    return args


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


def construct_queue(args, train_loader, tokenizer, key_encoder):
    flag = 0
    queue_num = int(args.queue_size * args.batch_size)
    queue = None
    while True:
        with torch.no_grad():
            for pid, data in enumerate(train_loader):
                # 和初始数据不同的数据作为反例
                if pid < 10:
                    continue
                query_encodings = tokenizer(data,
                                            padding=True,
                                            truncation=True,
                                            max_length=args.max_length,
                                            return_tensors='pt')
                query_encodings = {key: value.to(args.device) for key, value in query_encodings.items()}
                query_embedding = key_encoder(**query_encodings)
                if queue is None:
                    queue = query_embedding
                else:
                    if queue.shape[0] < queue_num:
                        queue = torch.cat((queue, query_embedding), 0)
                    else:
                        flag = 1
                if flag == 1:
                    break
        if flag == 1:
            break
    queue = queue[-queue_num:]
    queue = torch.div(queue, torch.norm(queue, dim=1).reshape(-1, 1))
    return queue
    

def repeat_word(tokenizer, sentence, dup_rate):
    '''
    @function: 重复句子中的部分token
    
    @input:
    sentence: string，输入语句
    
    @return:
    dup_sentence: string，重复token后生成的句子
    '''
    word_tokens = tokenizer.tokenize(sentence)
    
    # dup_len ∈ [0, max(2, int(dup_rate ∗ N))]
    max_len = max(2, int(dup_rate * len(word_tokens)))
    # 防止随机挑选的数值大于token数量
    dup_len = min(random.choice(range(max_len+1)), len(word_tokens))
    
    random_indices = random.sample(range(len(word_tokens)), dup_len)
    # print(max_len, dup_len, random_indices)
    
    dup_word_tokens = []
    for index, word in enumerate(word_tokens):
        dup_word_tokens.append(word)
        if index in random_indices and "#" not in word:
            dup_word_tokens.append(word)
    dup_sentence = tokenizer.decode(tokenizer.convert_tokens_to_ids(dup_word_tokens)).replace(" ", "")
    # dup_sentence = "".join(dup_word_tokens)
    return dup_sentence


def eval(model, tokenizer, dev_loader, args):
    model.eval()
    model.to(args.device)
    
    all_sims, all_scores = [], []
    with torch.no_grad():
        for data in dev_loader:
            sent1 = data['sent1']
            sent2 = data['sent2']
            score = data['score']
            sent1_encoding = tokenizer(sent1,
                                       padding=True,
                                       truncation=True,
                                       max_length=args.max_length,
                                       return_tensors='pt')
            sent1_encoding = {key: value.to(args.device)  for key, value in sent1_encoding.items()}
            sent2_encoding = tokenizer(sent2,
                                       padding=True,
                                       truncation=True,
                                       max_length=args.max_length,
                                       return_tensors='pt')
            sent2_encoding = {key: value.to(args.device)  for key, value in sent2_encoding.items()}
            
            sent1_output = model(**sent1_encoding)
            sent2_output = model(**sent2_encoding)
            sim_score = F.cosine_similarity(sent1_output, sent2_output).cpu().tolist()
            all_sims += sim_score
            all_scores += score.tolist()
    corr = scipy.stats.spearmanr(all_sims, all_scores).correlation
    return corr


def train(args):
    train_file = args.trainfile
    dev_file = args.devfile
    test_file = args.testfile
    file_type = args.filetype
    queue_num = int(args.queue_size * args.batch_size)
    match_dataset = MatchingDataSet()
    train_list = match_dataset.read_train_file(train_file, dev_file, test_file, file_type)
    dev_list = match_dataset.read_eval_file(dev_file, file_type)
    logger.info("train samples num: {}, dev samples num: {}".format(len(train_list), len(dev_list)))

    train_loader = DataLoader(train_list,
                              batch_size=args.batch_size,
                              shuffle=True)
    dev_loader = DataLoader(dev_list,
                            batch_size=args.batch_size)
    logger.info("train batch num: {}, dev batch num: {}".format(len(train_loader), len(dev_loader)))
    
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    query_encoder = ESimCSE(args.pretrained, args.pool_type, args.dropout_rate)
    key_encoder = copy.deepcopy(query_encoder)
    query_encoder.train()
    query_encoder.to(args.device)
    key_encoder.eval()
    key_encoder.to(args.device)
    
    optimizer = torch.optim.AdamW(query_encoder.parameters(), lr=args.lr)
    # 构造反例样本队列
    queue_embeddings = construct_queue(args, train_loader, tokenizer, key_encoder)
    
    batch_idx = 0
    best_corr = 0
    best_loss = 1000000
    for epoch in range(args.epochs):
        epoch_losses = []
        for data in train_loader:
            optimizer.zero_grad()
            # 构造正例样本
            key_data = [repeat_word(tokenizer, sentence, args.dup_rate) for sentence in data]

            query_encodings = tokenizer(data,
                                        padding=True,
                                        truncation=True,
                                        max_length=args.max_length,
                                        return_tensors='pt')
            query_encodings = {key: value.to(args.device) for key, value in query_encodings.items()}
            key_encodings = tokenizer(key_data,
                                      padding=True,
                                      truncation=True,
                                      max_length=args.max_length,
                                      return_tensors='pt')
            key_encodings = {key: value.to(args.device) for key, value in key_encodings.items()}
            
            query_embeddings = query_encoder(**query_encodings)
            key_embeddings = key_encoder(**key_encodings).detach()
            
            # 对表征进行归一化，便于后面相似度计算
            query_embeddings = F.normalize(query_embeddings, dim=1)
            key_embeddings = F.normalize(key_embeddings, dim=1)
            
            batch_loss = compute_loss(query_embeddings, key_embeddings, queue_embeddings, args.tao)
            epoch_losses.append(batch_loss.item())
            # print(batch_idx, batch_loss.item())
            
            batch_loss.backward()
            optimizer.step()
            
            # 更新队列中负样本表征
            # queue_embeddings = torch.cat((queue_embeddings, query_embeddings.detach()), 0)
            queue_embeddings = torch.cat((queue_embeddings, key_embeddings), 0)
            queue_embeddings = queue_embeddings[-queue_num:, :]
            
            # 更新key编码器的动量
            for query_params, key_params in zip(query_encoder.parameters(), key_encoder.parameters()):
                key_params.data.copy_(args.momentum * key_params + (1-args.momentum) * query_params)
                key_params.requires_grad = False
                
            if batch_idx % args.display_interval == 0:
                logger.info("Epoch: {}, batch: {}, loss: {}".format(epoch, batch_idx, batch_loss.item()))
            batch_idx += 1
            
        avg_epoch_loss = np.mean(epoch_losses)
        dev_corr = eval(query_encoder, tokenizer, dev_loader, args)
        logger.info("epoch: {}, avg loss: {}, dev corr: {}".format(epoch, avg_epoch_loss, dev_corr))
        # if avg_epoch_loss <= best_loss and dev_corr >= best_corr:
        if dev_corr >= best_corr:
            best_corr = dev_corr
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch,
                'batch': batch_idx,
                'model_state_dict': query_encoder.state_dict(),
                'loss': best_loss,
                'corr': best_corr
            }, args.model_out)
            logger.info("epoch: {}, batch: {}, best loss: {}, best corr: {}, save model".format(epoch, batch_idx, avg_epoch_loss, dev_corr))
            
        
if __name__ == "__main__":
    args = parse_args()
    logger.info("args: {}".format(args))
    train(args)
