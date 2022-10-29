# -*- coding: utf-8 -*-
# @Time    : 2022/9/17
# @Author  : Maciel

import torch.nn as nn
from transformers import AutoConfig, AutoModel
import torch


class ESimCSE(nn.Module):
    def __init__(self, pretrained="hfl/chinese-bert-wwm-ext", pool_type="cls", dropout_prob=0.3):
        super().__init__()
        conf = AutoConfig.from_pretrained(pretrained)
        conf.attention_probs_dropout_prob = dropout_prob
        conf.hidden_dropout_prob = dropout_prob
        self.encoder = AutoModel.from_pretrained(pretrained, config=conf)
        assert pool_type in ["cls", "pooler", "avg_first_last", "avg_last_two"], "invalid pool_type: %s" % pool_type
        self.pool_type = pool_type


    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.encoder(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              output_hidden_states=True)
        hidden_states = output.hidden_states
        if self.pool_type == "cls":
            output = output.last_hidden_state[:, 0]
        elif self.pool_type == "pooler":
            output = output.pooler_output
        elif self.pool_type == "avg_first_last":
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