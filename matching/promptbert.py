from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
from config import Params


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
    
    
if __name__ == "__main__":
    prompt_templates = '[UNK]，这句话的意思是[MASK]'
    # sentence = "天气很好"
    sentence = '"[X][X][X]"，这句话的意思是[MASK]。'
    tokenizer = AutoTokenizer.from_pretrained(Params.pretrained_model_path)
    special_token_dict = {'additional_special_tokens': ['[X]']}
    tokenizer.add_special_tokens(special_token_dict)
    sen_tokens = tokenizer.tokenize(sentence)
    sen_encodings = tokenizer(sentence,
                              return_tensors='pt')
    print(sen_tokens)
    print(sen_encodings)
    exit(0)
    
    mask_id = tokenizer.convert_tokens_to_ids(Params.mask_token)
    
    model = PromptBERT(Params.pretrained_model_path, Params.dropout, mask_id)
    model.train()
    
    while True:
        print("input your sentence:")
        sentences = input()
        sentence_list = sentences.split(";")
    
        prompt_lines, template_lines = [], []
        for sentence in sentence_list:
            words_num = len(tokenizer.tokenize(sentence))
            prompt_line = prompt_templates.replace('[UNK]', sentence)
            template_line = prompt_templates.replace('[UNK]', '[UNK]'*words_num)
            print("prompt_line: {}, template_line: {}".format(prompt_line, template_line))
            prompt_lines.append(prompt_line)
            template_lines.append(template_line)
        
        prompt_encodings = tokenizer(list(prompt_lines),
                                            padding=True,
                                            truncation=True,
                                            max_length=Params.max_length,
                                            return_tensors='pt')
        template_encodings = tokenizer(list(template_lines),
                                        padding=True,
                                        truncation=True,
                                        max_length=Params.max_length,
                                        return_tensors='pt')
        
        prompt_mask_embedding = model.calculate_mask_embedding(prompt_encodings['input_ids'],
                                                               prompt_encodings['attention_mask'],
                                                               prompt_encodings['token_type_ids'])
        template_mask_embedding = model.calculate_mask_embedding(template_encodings['input_ids'],
                                                                 template_encodings['attention_mask'],
                                                                 template_encodings['token_type_ids'])
        # print(prompt_mask_embedding.shape)
        