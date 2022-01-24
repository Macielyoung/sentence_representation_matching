# -*- coding: utf-8 -*-
# @Time    : 2021/12/16
# @Author  : Maciel

import torch.nn as nn
from transformers import BertConfig, BertModel


class ESimCSE(nn.Module):
    def __init__(self, pretrained="hfl/chinese-bert-wwm-ext", dropout_prob=0.15):
        super().__init__()
        conf = BertConfig.from_pretrained(pretrained)
        conf.attention_probs_dropout_prob = dropout_prob
        conf.hidden_dropout_prob = dropout_prob
        self.encoder = BertModel.from_pretrained(pretrained, config=conf)
        self.fc = nn.Linear(conf.hidden_size, conf.hidden_size)


    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.encoder(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        output = output.last_hidden_state[:, 0]
        output = self.fc(output)
        return output