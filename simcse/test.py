# -*- coding: utf-8 -*-
# @Time    : 2022/9/19
# @Author  : Maciel

from loading import MatchingDataSet
import torch
import torch.nn.functional as F
import scipy.stats
from torch.utils.data import DataLoader
import numpy as np
from SimCSE import SimCSE
from transformers import AutoTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def eval(model, tokenizer, test_loader, device, max_length):
    model.eval()
    model.to(device)
    
    all_sims, all_scores = [], []
    with torch.no_grad():
        for data in test_loader:
            sent1 = data['sent1']
            sent2 = data['sent2']
            score = data['score']
            sent1_encoding = tokenizer(sent1,
                                       padding=True,
                                       truncation=True,
                                       max_length=max_length,
                                       return_tensors='pt')
            sent1_encoding = {key: value.to(device)  for key, value in sent1_encoding.items()}
            sent2_encoding = tokenizer(sent2,
                                       padding=True,
                                       truncation=True,
                                       max_length=max_length,
                                       return_tensors='pt')
            sent2_encoding = {key: value.to(device)  for key, value in sent2_encoding.items()}
            
            sent1_output = model(**sent1_encoding)
            sent2_output = model(**sent2_encoding)
            sim_score = F.cosine_similarity(sent1_output, sent2_output).cpu().tolist()
            all_sims += sim_score
            all_scores += score.tolist()
    corr = scipy.stats.spearmanr(all_sims, all_scores).correlation
    return corr


def test(testfile, pretrained, pool_type, dropout_rate, model_path, max_length):
    match_dataset = MatchingDataSet()
    testfile_type = "txt"
    test_list = match_dataset.read_eval_file(testfile, testfile_type)
    print("test samples num: {}".format(len(test_list)))

    test_loader = DataLoader(test_list,
                             batch_size=128)
    print("test batch num: {}".format(len(test_loader)))
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = SimCSE(pretrained, pool_type, dropout_rate)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_corr = eval(model, tokenizer, test_loader, device, max_length)
    print("test corr: {}".format(test_corr))
    

if __name__ == "__main__":
    testfile = "../dataset/STS-B/test.txt"
    pretrained = "hfl/chinese-roberta-wwm-ext-large"    
    pool_type = "avg_first_last"
    dropout_rate = 0.3
    max_length = 128
    model_path = "../models/simcse_roberta_large_stsb.pth"
    
    test(testfile, pretrained, pool_type, dropout_rate, model_path, max_length)
