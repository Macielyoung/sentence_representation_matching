# -*- coding: utf-8 -*-
# @Time    : 2022/9/21
# @Author  : Maciel

import torch.nn as nn
from transformers import AutoConfig, AutoModel
import torch


class PromptBERT(nn.Module):
    def __init__(self, pretrained_model_path, dropout_prob, mask_id):
        super().__init__()
        conf = AutoConfig.from_pretrained(pretrained_model_path)
        conf.attention_probs_dropout_prob = dropout_prob
        conf.hidden_dropout_prob = dropout_prob
        self.encoder = AutoModel.from_pretrained(pretrained_model_path, config=conf)
        self.mask_id = mask_id

        
    def forward(self, prompt_input_ids, prompt_attention_mask, prompt_token_type_ids, template_input_ids, template_attention_mask, template_token_type_ids):
        '''
        @function: 计算prompt mask标签表征向量hi和模板向量表征h^i之间的差
        
        @input:
        prompt_input_ids:          prompt句子输入id
        prompt_attention_mask:     prompt句子注意力矩阵
        prompt_token_type_ids:     prompt句子token类型id
        template_input_ids:        模板句子输入id
        template_attention_mask:   模板句子注意力矩阵
        template_token_type_ids:   模板句子token类型id
        
        @return: sentence_embedding: 句子表征向量
        '''
        prompt_mask_embedding = self.calculate_mask_embedding(prompt_input_ids, prompt_attention_mask, prompt_token_type_ids)
        template_mask_embedding = self.calculate_mask_embedding(template_input_ids, template_attention_mask, template_token_type_ids)
        sentence_embedding = prompt_mask_embedding - template_mask_embedding
        return sentence_embedding
        
        
    def calculate_mask_embedding(self, input_ids, attention_mask, token_type_ids):
        # print("input_ids: ", input_ids)
        output = self.encoder(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        token_embeddings = output[0]
        mask_index = (input_ids == self.mask_id).long()
        # print("mask_index: ", mask_index)
        mask_embedding = self.get_mask_embedding(token_embeddings, mask_index)
        return mask_embedding
    
    
    def get_mask_embedding(self, token_embeddings, mask_index):
        '''
        @function: 获取[mask]标签的embedding输出
        
        @input:
        token_embeddings: Tensor, 编码层最后一层token输出
        mask_index:   Tensor, mask标签位置
        
        @return: mask_embedding: Tensor, mask标签embedding
        '''
        input_mask_expanded = mask_index.unsqueeze(-1).expand(token_embeddings.size()).float()
        # print("input_mask_expanded: ", input_mask_expanded)
        # print("input mask expaned shape: ", input_mask_expanded.shape)
        mask_embedding = torch.sum(token_embeddings * input_mask_expanded, 1)
        return mask_embedding
    
    
    def forward_sentence(self, input_ids, attention_mask, token_type_ids, pool_type):
        output = self.encoder(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              output_hidden_states=True)
        hidden_states = output.hidden_states
        if pool_type == "cls":
            output = output.last_hidden_state[:, 0]
        elif pool_type == "pooler":
            output = output.pooler_output
        elif pool_type == "avg_first_last":
            top_first_state = self.get_avg_tensor(hidden_states[1], attention_mask)
            last_first_state = self.get_avg_tensor(hidden_states[-1], attention_mask)
            output = (top_first_state + last_first_state) / 2
        else:
            last_first_state = self.get_avg_tensor(hidden_states[-1], attention_mask)
            last_second_state = self.get_avg_tensor(hidden_states[-2], attention_mask)
            output = (last_first_state + last_second_state) / 2
        return output
    
    
    def get_avg_tensor(self, layer_hidden_state, attention_mask):
        '''
        layer_hidden_state: 模型一层表征向量 [B * L * D]
        attention_mask: 句子padding位置 [B * L]
        return: avg_embeddings, 非零位置词语的平均向量 [B * D]
        '''
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden_state.size()).float()
        sum_embeddings = torch.sum(layer_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        avg_embeddings = sum_embeddings / sum_mask
        return avg_embeddings