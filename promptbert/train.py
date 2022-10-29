# -*- coding: utf-8 -*-
# @Time    : 2022/9/21
# @Author  : Maciel


from transformers import AutoTokenizer
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import scipy.stats
from loading import MatchingDataSet
from PromptBert import PromptBERT
import argparse
from loguru import logger
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trainfile", type=str, default="../dataset/STS-B/train.txt", help="train file path")
    parser.add_argument("--devfile", type=str, default="../dataset/STS-B/dev.txt", help="dev file path")
    parser.add_argument("--testfile", type=str, default="../dataset/STS-B/test.txt", help="test file path")
    parser.add_argument("--filetype", type=str, default="txt", help="train and dev file type")
    parser.add_argument("--pretrained", type=str, default="hfl/chinese-roberta-wwm-ext", help="huggingface pretrained model")
    parser.add_argument("--model_out", type=str, default="../models/promptbert_roberta_stsb.pth", help="model output path")
    parser.add_argument("--max_length", type=int, default=128, help="sentence max length")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=10, help="epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--tao", type=float, default=0.05, help="temperature")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--display_interval", type=int, default=100, help="display interval")
    parser.add_argument("--pool_type", type=str, default="avg_first_last", help="pool_type")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="dropout_rate")
    parser.add_argument("--task", type=str, default="promptbert", help="task name")
    args = parser.parse_args()
    return args


def compute_loss(query, key, tao=0.05):
    # 对表征进行归一化，便于后面相似度计算
    query = F.normalize(query, dim=1)
    key = F.normalize(key, dim=1)
    # print(query.shape, key.shape)
    N, D = query.shape[0], query.shape[1]
    
    # calculate positive similarity
    batch_pos = torch.exp(torch.div(torch.bmm(query.view(N, 1, D), key.view(N, D, 1)).view(N, 1), tao))
    
    # calculate inner_batch all similarity
    batch_all = torch.sum(torch.exp(torch.div(torch.mm(query.view(N, D), torch.t(key)), tao)), dim=1)
    
    loss = torch.mean(-torch.log(torch.div(batch_pos, batch_all)))
    return loss


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
            
            sent1_output = model.forward_sentence(sent1_encoding['input_ids'],
                                                  sent1_encoding['attention_mask'],
                                                  sent1_encoding['token_type_ids'],
                                                  args.pool_type)
            sent2_output = model.forward_sentence(sent2_encoding['input_ids'],
                                                  sent2_encoding['attention_mask'],
                                                  sent2_encoding['token_type_ids'],
                                                  args.pool_type)
            sim_score = F.cosine_similarity(sent1_output, sent2_output).cpu().tolist()
            all_sims += sim_score
            all_scores += score.tolist()
    corr = scipy.stats.spearmanr(all_sims, all_scores).correlation
    return corr


def eval2(model, tokenizer, dev_loader, args):
    model.eval()
    model.to(args.device)
    
    all_sims, all_scores = [], []
    with torch.no_grad():
        for data in dev_loader:
            sent1 = data['sent1']
            sent2 = data['sent2']
            score = data['score']
            
            prompt_template_sent1 = [transform_sentence(s1, tokenizer, args) for s1 in sent1]
            prompt_sent1 = [pair[0] for pair in prompt_template_sent1]
            template_sent1 = [pair[1] for pair in prompt_template_sent1]
            prompt_encoding1 = encode_sentences(tokenizer, prompt_sent1, args)
            template_encoding1 = encode_sentences(tokenizer, template_sent1, args)
            prompt_encoding1 = {key: value.to(args.device) for key, value in prompt_encoding1.items()}
            template_encoding1 = {key: value.to(args.device) for key, value in template_encoding1.items()}
            
            prompt_template_sent2 = [transform_sentence(s2, tokenizer, args) for s2 in sent2]
            prompt_sent2 = [pair[0] for pair in prompt_template_sent2]
            template_sent2 = [pair[1] for pair in prompt_template_sent2]
            prompt_encoding2 = encode_sentences(tokenizer, prompt_sent2, args)
            template_encoding2 = encode_sentences(tokenizer, template_sent2, args)
            prompt_encoding2 = {key: value.to(args.device) for key, value in prompt_encoding2.items()}
            template_encoding2 = {key: value.to(args.device) for key, value in template_encoding2.items()}
            
            sent1_output = model(prompt_encoding1['input_ids'],
                                 prompt_encoding1['attention_mask'],
                                 prompt_encoding1['token_type_ids'],
                                 template_encoding1['input_ids'],
                                 template_encoding1['attention_mask'],
                                 template_encoding1['token_type_ids'])
            sent2_output = model(prompt_encoding2['input_ids'],
                                 prompt_encoding2['attention_mask'],
                                 prompt_encoding2['token_type_ids'],
                                 template_encoding2['input_ids'],
                                 template_encoding2['attention_mask'],
                                 template_encoding2['token_type_ids'])
            sim_score = F.cosine_similarity(sent1_output, sent2_output).cpu().tolist()
            all_sims += sim_score
            all_scores += score.tolist()
    corr = scipy.stats.spearmanr(all_sims, all_scores).correlation
    return corr


def transform_sentence(sentence, tokenizer, args):
    prompt_templates = ['[X]，它的意思是[MASK]', '[X]，这句话的意思是[MASK]']
    words_list = tokenizer.tokenize(sentence)
    words_num = len(words_list)
    sentence_template = []
    for template in prompt_templates:
        if words_num > args.max_length - 15:
            words_list = words_list[:-15]
            sentence = tokenizer.decode(tokenizer.convert_tokens_to_ids(words_list)).replace(" ", "")
        
        words_len = len(tokenizer.tokenize(sentence))
        prompt_sentence = template.replace("[X]", sentence)
        template_sentence = template.replace("[X]", "[X]"*words_len)
        sentence_template += [prompt_sentence, template_sentence]
    return sentence_template


def encode_sentences(tokenizer, sen_list, args):
    sen_encoding = tokenizer(sen_list,
                             padding=True,
                             truncation=True,
                             max_length=args.max_length,
                             return_tensors='pt')
    return sen_encoding


def build_dataset(dataloader, tokenizer, args):
    data_encodings = []
    for data in dataloader:
        prompt_template_sentences = [transform_sentence(sentence, tokenizer, args) for sentence in data]
        prompt_sent1_list = [pair[0] for pair in prompt_template_sentences]
        template_sent1_list = [pair[1] for pair in prompt_template_sentences]
        prompt_sent2_list = [pair[2] for pair in prompt_template_sentences]
        template_sent2_list = [pair[3] for pair in prompt_template_sentences]
            
        prompt_encoding1 = encode_sentences(tokenizer, prompt_sent1_list, args)
        template_encoding1 = encode_sentences(tokenizer, template_sent1_list, args)
        prompt_encoding2 = encode_sentences(tokenizer, prompt_sent2_list, args)
        template_encoding2 = encode_sentences(tokenizer, template_sent2_list, args)
        data_encodings.append([prompt_encoding1, template_encoding1, prompt_encoding2, template_encoding2])
    return data_encodings
    

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
    special_token_dict = {'additional_special_tokens': ['[X]']}
    tokenizer.add_special_tokens(special_token_dict)
    mask_id = tokenizer.mask_token_id
    
    model = PromptBERT(args.pretrained, args.dropout_rate, mask_id)
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.train()
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    train_encodings = build_dataset(train_loader, tokenizer, args)
    
    batch_idx = 0
    best_corr = 0
    best_loss = 1000000
    for epoch in range(args.epochs):
        epoch_losses = []
        for data in train_encodings:
            optimizer.zero_grad()
            prompt_encoding1, template_encoding1, prompt_encoding2, template_encoding2 = data
            prompt_encoding1 = {key: value.to(args.device) for key, value in prompt_encoding1.items()}
            template_encoding1 = {key: value.to(args.device) for key, value in template_encoding1.items()}
            prompt_encoding2 = {key: value.to(args.device) for key, value in prompt_encoding2.items()}
            template_encoding2 = {key: value.to(args.device) for key, value in template_encoding2.items()}
            
            query_embedding = model(prompt_encoding1['input_ids'],
                                    prompt_encoding1['attention_mask'],
                                    prompt_encoding1['token_type_ids'],
                                    template_encoding1['input_ids'],
                                    template_encoding1['attention_mask'],
                                    template_encoding1['token_type_ids'])
            key_embedding = model(prompt_encoding2['input_ids'],
                                  prompt_encoding2['attention_mask'],
                                  prompt_encoding2['token_type_ids'],
                                  template_encoding2['input_ids'],
                                  template_encoding2['attention_mask'],
                                  template_encoding2['token_type_ids'])
            batch_loss = compute_loss(query_embedding, key_embedding, args.tao)
            
            batch_loss.backward()
            optimizer.step()
            epoch_losses.append(batch_loss.item())
            if batch_idx % args.display_interval == 0:
                logger.info("Epoch: {}, batch: {}, loss: {}".format(epoch, batch_idx, batch_loss.item()))
            batch_idx += 1
            
        avg_epoch_loss = np.mean(epoch_losses)
        dev_corr = eval2(model, tokenizer, dev_loader, args)
        logger.info("epoch: {}, avg loss: {}, dev corr: {}".format(epoch, avg_epoch_loss, dev_corr))
        if avg_epoch_loss <= best_loss and dev_corr >= best_corr:
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
    logger.info("args: {}".format(args))
    train(args)
    
    # sentence = "今天天气真不错啊"
    # tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    # special_token_dict = {'additional_special_tokens': ['[X]']}
    # tokenizer.add_special_tokens(special_token_dict)
    # sentence_template = transform_sentence(sentence, tokenizer, args)
    # print(sentence_template)

    # st_encoding = encode_sentences(tokenizer, sentence_template, args)
    # print(st_encoding['input_ids'])
