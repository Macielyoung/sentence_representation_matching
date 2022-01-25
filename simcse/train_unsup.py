# -*- coding: utf-8 -*-
# @Time    : 2021/6/10
# @Author  : kaka


import argparse
import logging
import os
from config import Params

from datasets import load_dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
from SimCSE import SimCSE
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("train_file", type=str, help="train text file")
    # parser.add_argument("--pretrained", type=str, default="hfl/chinese-bert-wwm-ext", help="huggingface pretrained model")
    # parser.add_argument("--model_out", type=str, default="./finder_model", help="model output path")
    parser.add_argument("--num_proc", type=int, default=1, help="dataset process thread num")
    parser.add_argument("--max_length", type=int, default=64, help="sentence max length")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=101, help="epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--tao", type=float, default=0.05, help="temperature")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--display_interval", type=int, default=500, help="display interval")
    parser.add_argument("--save_interval", type=int, default=10, help="save interval")
    parser.add_argument("--pool_type", type=str, default="pooler", help="pool_type")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="dropout_rate")
    args = parser.parse_args()
    return args


def read_data(args):
    with open(Params.dialogues_file, 'r') as f:
        sentences = f.readlines()
    dl = DataLoader(sentences,
                    batch_size=args.batch_size)
    return dl


def duplicate_batch(batch, tokenzier, args):
    '''
    句子进行重复
    '''
    new_batch = []
    for sentence in batch:
        new_batch.append(sentence)
        new_batch.append(sentence)
    batch_encoding = tokenzier(new_batch,
                               padding=True,
                               truncation=True,
                               max_length=args.max_length,
                               return_tensors='pt')
    return batch_encoding


def compute_loss(y_pred, tao=0.05, device="cuda"):
    idxs = torch.arange(0, y_pred.shape[0], device=device)
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
    similarities = similarities / tao
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(Params.pretrained_model_path)
    dl = read_data(args)
    model = SimCSE(Params.pretrained_model_path, args.pool_type, args.dropout_rate).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    batch_idx = 0
    min_loss = 10000000
    for epoch_idx in range(args.epochs):
        epoch_losses = []
        for data in tqdm(dl):
            batch_idx += 1
            new_batch_data = duplicate_batch(data, tokenizer, args)
            pred = model(input_ids=new_batch_data["input_ids"].to(args.device),
                         attention_mask=new_batch_data["attention_mask"].to(args.device),
                         token_type_ids=new_batch_data["token_type_ids"].to(args.device))
            loss = compute_loss(pred, args.tao, args.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            epoch_losses.append(loss)
            if batch_idx % args.display_interval == 0:
                logging.info(f"epoch: {epoch_idx}, batch_idx: {batch_idx}, loss: {loss:>10f}")
        avg_epoch_loss = np.mean(epoch_losses)
        if avg_epoch_loss < min_loss:
            min_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch_idx,
                'model_state_dict': model.state_dict(),
                'loss': avg_epoch_loss
            }, Params.simcse_model_path)
        

def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
