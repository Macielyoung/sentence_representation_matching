# -*- coding: utf-8 -*-
# @Time    : 2022/1/5
# @Author  : Maciel

import torch.nn as nn
from transformers import BertConfig, BertModel
import torch


class SBERT(nn.Module):
    def __init__(self, pretrained="hfl/chinese-bert-wwm-ext", pool_type="cls", dropout_prob=0.3):
        super().__init__()
        conf = BertConfig.from_pretrained(pretrained)
        conf.attention_probs_dropout_prob = dropout_prob
        conf.hidden_dropout_prob = dropout_prob
        self.encoder = BertModel.from_pretrained(pretrained, config=conf)
        assert pool_type in ["cls", "pooler", "mean"], "invalid pool_type: %s" % pool_type
        self.pool_type = pool_type


    def forward(self, input_ids, attention_mask, token_type_ids):
        if self.pool_type == "cls":
            output = self.encoder(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
            output = output.last_hidden_state[:, 0]
        elif self.pool_type == "pooler":
            output = self.encoder(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
            output = output.pooler_output
        elif self.pool_type == "mean":
            output = self.get_mean_tensor(input_ids, attention_mask)
        return output
    
    
    def get_mean_tensor(self, input_ids, attention_mask):
        encode_states = self.encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = encode_states.hidden_states
        last_avg_state = self.get_avg_tensor(hidden_states[-1], attention_mask)
        first_avg_state = self.get_avg_tensor(hidden_states[1], attention_mask)
        mean_avg_state = (last_avg_state + first_avg_state) / 2
        return mean_avg_state
    
    
    def get_avg_tensor(self, layer_hidden_state, attention_mask):
        '''
        layer_hidden_state: 模型一层表征向量 [B * L * D]
        attention_mask: 句子padding位置 [B * L]
        return: 非零位置词语的平均向量 [B * D]
        '''
        layer_hidden_dim = layer_hidden_state.shape[-1]
        attention_repeat_mask = attention_mask.unsqueeze(dim=-1).tile(layer_hidden_dim)
        layer_attention_state = torch.mul(layer_hidden_state, attention_repeat_mask)
        layer_sum_state = layer_attention_state.sum(dim=1)
        # print(last_attention_state.shape)
        
        attention_length_mask = attention_mask.sum(dim=-1)
        attention_length_repeat_mask = attention_length_mask.unsqueeze(dim=-1).tile(layer_hidden_dim)
        # print(attention_length_repeat_mask.shape)
        
        layer_avg_state = torch.mul(layer_sum_state, 1/attention_length_repeat_mask)
        return layer_avg_state
    
    
    def get_avg_tensor2(self, layer_hidden_state, attention_mask):
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