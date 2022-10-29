# -*- coding: utf-8 -*-
# @Time    : 2022/9/17
# @Author  : Maciel


import argparse
import os
from loading import MatchingDataSet
import torch
import torch.nn.functional as F
import scipy.stats
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
from SimCSE import SimCSE
from loguru import logger
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logger.add("../runtime.log")


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trainfile", type=str, default="../dataset/STS-B/train.txt", help="train file path")
    parser.add_argument("--devfile", type=str, default="../dataset/STS-B/dev.txt", help="dev file path")
    parser.add_argument("--testfile", type=str, default="../dataset/STS-B/test.txt", help="test file path")
    parser.add_argument("--filetype", type=str, default="txt", help="train and dev file type")
    parser.add_argument("--pretrained", type=str, default="hfl/chinese-roberta-wwm-ext-large", help="huggingface pretrained model")
    parser.add_argument("--model_out", type=str, default="../models/simcse_roberta_large_stsb.pth", help="model output path")
    # parser.add_argument("--num_proc", type=int, default=1, help="dataset process thread num")
    parser.add_argument("--max_length", type=int, default=128, help="sentence max length")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--tao", type=float, default=0.05, help="temperature")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--display_interval", type=int, default=100, help="display interval")
    # parser.add_argument("--save_interval", type=int, default=10, help="save interval")
    parser.add_argument("--pool_type", type=str, default="avg_first_last", help="pool_type")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="dropout_rate")
    parser.add_argument("--task", type=str, default="simcse", help="task name")
    args = parser.parse_args()
    return args


def duplicate_batch(batch):
    '''
    重复两次数据
    '''
    new_batch = []
    for sentence in batch:
        new_batch += [sentence, sentence]
    return new_batch


def compute_loss(y_pred, tao=0.05, device="cuda"):
    idxs = torch.arange(0, y_pred.shape[0], device=device)
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
    similarities = similarities / tao
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


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
    model = SimCSE(args.pretrained, args.pool_type, args.dropout_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    model.to(args.device)
    
    batch_idx = 0
    best_corr = 0
    best_loss = 1000000
    for epoch in range(args.epochs):
        epoch_losses = []
        for data in train_loader:
            batch_idx += 1
            batch_data = duplicate_batch(data)
            encodings = tokenizer(batch_data,
                                  padding=True,
                                  truncation=True,
                                  max_length=args.max_length,
                                  return_tensors='pt')
            encodings = {key: value.to(args.device)  for key, value in encodings.items()}
            output = model(**encodings)
            batch_loss = compute_loss(output, args.tao, args.device)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_losses.append(batch_loss.item())
            if batch_idx % args.display_interval == 0:
                logger.info("epoch: {}, batch: {}, loss: {}".format(epoch, batch_idx, batch_loss.item()))
        avg_epoch_loss = np.mean(epoch_losses)
        dev_corr = eval(model, tokenizer, dev_loader, args)
        logger.info("epoch: {}, avg loss: {}, dev corr: {}".format(epoch, avg_epoch_loss, dev_corr))
        if dev_corr >= best_corr and avg_epoch_loss <= best_loss:
            best_corr = dev_corr
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch,
                'batch': batch_idx,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'corr': best_corr
            }, args.model_out)
            logger.info("epoch: {}, batch: {}, best loss: {}, best corr: {}, save model".format(epoch, batch_idx, avg_epoch_loss, dev_corr))
            
            
if __name__ == "__main__":
    args = parse_args()
    train(args)
