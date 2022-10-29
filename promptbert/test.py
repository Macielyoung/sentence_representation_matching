# -*- coding: utf-8 -*-
# @Time    : 2022/9/19
# @Author  : Maciel

from loading import MatchingDataSet
import torch
import torch.nn.functional as F
import scipy.stats
from torch.utils.data import DataLoader
from PromptBert import PromptBERT
from transformers import AutoTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def eval(model, tokenizer, dev_loader, device, max_length, pool_type):
    model.eval()
    model.to(device)
    
    all_sims, all_scores = [], []
    with torch.no_grad():
        for data in dev_loader:
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
            
            sent1_output = model.forward_sentence(sent1_encoding['input_ids'],
                                                  sent1_encoding['attention_mask'],
                                                  sent1_encoding['token_type_ids'],
                                                  pool_type)
            sent2_output = model.forward_sentence(sent2_encoding['input_ids'],
                                                  sent2_encoding['attention_mask'],
                                                  sent2_encoding['token_type_ids'],
                                                  pool_type)
            sim_score = F.cosine_similarity(sent1_output, sent2_output).cpu().tolist()
            all_sims += sim_score
            all_scores += score.tolist()
    corr = scipy.stats.spearmanr(all_sims, all_scores).correlation
    return corr


def eval2(model, tokenizer, dev_loader, device, max_length, pool_type):
    model.eval()
    model.to(device)
    
    all_sims, all_scores = [], []
    with torch.no_grad():
        for data in dev_loader:
            sent1 = data['sent1']
            sent2 = data['sent2']
            score = data['score']
            
            # print("sent1: {}, sent2: {}".format(sent1, sent2))
            prompt_template_sent1 = [transform_sentence(s1, tokenizer, max_length) for s1 in sent1]
            prompt_sent1 = [pair[0] for pair in prompt_template_sent1]
            template_sent1 = [pair[1] for pair in prompt_template_sent1]
            # print("prompt sent1: {}".format(prompt_sent1))
            # print("template sent1: {}".format(template_sent1))
            prompt_encoding1 = encode_sentences(tokenizer, prompt_sent1, max_length)
            template_encoding1 = encode_sentences(tokenizer, template_sent1, max_length)
            prompt_encoding1 = {key: value.to(device) for key, value in prompt_encoding1.items()}
            template_encoding1 = {key: value.to(device) for key, value in template_encoding1.items()}
            # print("prompt_encoding1 input_ids {}".format(prompt_encoding1['input_ids']))
            # print("template_encoding1 input_ids: {}".format(template_encoding1['input_ids']))
            
            prompt_template_sent2 = [transform_sentence(s2, tokenizer, max_length) for s2 in sent2]
            prompt_sent2 = [pair[0] for pair in prompt_template_sent2]
            template_sent2 = [pair[1] for pair in prompt_template_sent2]
            # print("prompt sent2: {}".format(prompt_sent2))
            # print("template sent2: {}".format(template_sent2))
            prompt_encoding2 = encode_sentences(tokenizer, prompt_sent2, max_length)
            template_encoding2 = encode_sentences(tokenizer, template_sent2, max_length)
            prompt_encoding2 = {key: value.to(device) for key, value in prompt_encoding2.items()}
            template_encoding2 = {key: value.to(device) for key, value in template_encoding2.items()}
            # print("prompt_encoding2 input_ids: {}".format(prompt_encoding2['input_ids']))
            # print("template_encoding2 input_ids: {}".format(template_encoding2['input_ids']))
            
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
            # print("sen1 output shape: {}, sen2 output shape: {}".format(sent1_output.shape, sent2_output.shape))
            sim_score = F.cosine_similarity(sent1_output, sent2_output).cpu().tolist()
            all_sims += sim_score
            all_scores += score.tolist()
            # print("sim_score: {}, score: {}".format(sim_score, score))
    corr = scipy.stats.spearmanr(all_sims, all_scores).correlation
    return corr


def transform_sentence(sentence, tokenizer, max_length):
    prompt_templates = ['[X]，它的意思是[MASK]', '[X]，这句话的意思是[MASK]']
    words_list = tokenizer.tokenize(sentence)
    words_num = len(words_list)
    sentence_template = []
    for template in prompt_templates:
        if words_num > max_length - 15:
            words_list = words_list[:-15]
            sentence = tokenizer.decode(tokenizer.convert_tokens_to_ids(words_list)).replace(" ", "")
        
        words_len = len(tokenizer.tokenize(sentence))
        prompt_sentence = template.replace("[X]", sentence)
        template_sentence = template.replace("[X]", "[X]"*words_len)
        sentence_template += [prompt_sentence, template_sentence]
    return sentence_template


def encode_sentences(tokenizer, sen_list, max_length):
    sen_encoding = tokenizer(sen_list,
                             padding=True,
                             truncation=True,
                             max_length=max_length,
                             return_tensors='pt')
    return sen_encoding


def test(testfile, pretrained, pool_type, dropout_rate, model_path, max_length):
    match_dataset = MatchingDataSet()
    testfile_type = "txt"
    test_list = match_dataset.read_eval_file(testfile, testfile_type)
    print("test samples num: {}".format(len(test_list)))

    test_loader = DataLoader(test_list,
                             batch_size=4)
    print("test batch num: {}".format(len(test_loader)))
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    special_token_dict = {'additional_special_tokens': ['[X]']}
    tokenizer.add_special_tokens(special_token_dict)
    mask_id = tokenizer.mask_token_id
    model = PromptBERT(pretrained, dropout_rate, mask_id)
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # test_corr = eval(model, tokenizer, test_loader, device, max_length, pool_type)
    test_corr2 = eval2(model, tokenizer, test_loader, device, max_length, pool_type)
    # print("test corr: {}, test_corr2: {}".format(test_corr, test_corr2))
    print(test_corr2)
    

if __name__ == "__main__":
    testfile = "../dataset/STS-B/test.txt"
    pretrained = "hfl/chinese-roberta-wwm-ext"
    pool_type = "avg_first_last"
    dropout_rate = 0.1
    max_length = 128
    model_path = "../models/promptbert_roberta_stsb.pth"
    
    test(testfile, pretrained, pool_type, dropout_rate, model_path, max_length)
