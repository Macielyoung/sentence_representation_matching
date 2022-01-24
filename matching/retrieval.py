from simcse import SimCSE
from esimcse import ESimCSE
from promptbert import PromptBERT
from sbert import SBERT
from cosent import CoSent
from config import Params
from log import logger
import torch
from transformers import AutoTokenizer


class SimCSERetrieval(object):
    def __init__(self, pretrained_model_path, simcse_path, pool_type, dropout):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        model = SimCSE(Params.pretrained_model, pool_type, dropout)
        self.checkpoint = torch.load(simcse_path, map_location='cpu')
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        self.model = model
        
        
    def print_checkpoint_info(self):
        loss = self.checkpoint['loss']
        epoch = self.checkpoint['epoch']
        model_info = {'loss': loss, 'epoch': epoch}
        return model_info
        
        
    def calculate_sentence_embedding(self, sentence):
        device = "cpu"
        input_encodings = self.tokenizer(sentence,
                                         padding=True,
                                         truncation=True,
                                         max_length=Params.max_length,
                                         return_tensors='pt')
        sentence_embedding = self.model(input_encodings['input_ids'].to(device),
                                        input_encodings['attention_mask'].to(device),
                                        input_encodings['token_type_ids'].to(device))
        return sentence_embedding
    
    
    def calculate_sentence_similarity(self, sentence1, sentence2):
        sentence1 = sentence1.strip()
        sentence2 = sentence2.strip()
        sentence1_embedding = self.calculate_sentence_embedding(sentence1)
        sentence2_embedding = self.calculate_sentence_embedding(sentence2)
        similarity = torch.cosine_similarity(sentence1_embedding, sentence2_embedding, dim=-1)
        similarity = float(similarity.item())
        return similarity


class ESimCSERetrieval(object):
    def __init__(self, pretrained_model_path, esimcse_path, dropout):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        model = ESimCSE(Params.pretrained_model, dropout)
        self.checkpoint = torch.load(esimcse_path, map_location='cpu')
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        self.model = model
        
        
    def print_checkpoint_info(self):
        loss = self.checkpoint['loss']
        epoch = self.checkpoint['epoch']
        model_info = {'loss': loss, 'epoch': epoch}
        return model_info
        
        
    def calculate_sentence_embedding(self, sentence):
        device = "cpu"
        input_encodings = self.tokenizer(sentence,
                                         padding=True,
                                         truncation=True,
                                         max_length=Params.max_length,
                                         return_tensors='pt')
        sentence_embedding = self.model(input_encodings['input_ids'].to(device),
                                        input_encodings['attention_mask'].to(device),
                                        input_encodings['token_type_ids'].to(device))
        return sentence_embedding
    
    
    def calculate_sentence_similarity(self, sentence1, sentence2):
        sentence1 = sentence1.strip()
        sentence2 = sentence2.strip()
        sentence1_embedding = self.calculate_sentence_embedding(sentence1)
        sentence2_embedding = self.calculate_sentence_embedding(sentence2)
        similarity = torch.cosine_similarity(sentence1_embedding, sentence2_embedding, dim=-1)
        similarity = float(similarity.item())
        return similarity
    
    
class PromptBertRetrieval(object):
    def __init__(self, pretrained_model_path, promptbert_path, dropout):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        special_token_dict = {'additional_special_tokens': ['[X]']}
        self.tokenizer.add_special_tokens(special_token_dict)
        mask_id = self.tokenizer.convert_tokens_to_ids(Params.mask_token)
        model = PromptBERT(pretrained_model_path, dropout, mask_id)
        model.encoder.resize_token_embeddings(len(self.tokenizer))
        checkpoint = torch.load(promptbert_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        self.checkpoint = checkpoint
        self.model = model
        
        
    def print_checkpoint_info(self):
        loss = self.checkpoint['loss']
        epoch = self.checkpoint['epoch']
        model_info = {'loss': loss, 'epoch': epoch}
        return model_info
    
        
    def calculate_sentence_mask_embedding(self, sentence):
        device = "cpu"
        prompt_sentence = Params.prompt_templates[0].replace("[X]", sentence)
        prompt_encodings = self.tokenizer(prompt_sentence,
                                          padding=True,
                                          truncation=True,
                                          max_length=Params.max_length,
                                          return_tensors='pt')
        sentence_mask_embedding = self.model.calculate_mask_embedding(prompt_encodings['input_ids'].to(device),
                                                                      prompt_encodings['attention_mask'].to(device),
                                                                      prompt_encodings['token_type_ids'].to(device))
        return sentence_mask_embedding
        
        
    def calculate_sentence_embedding(self, sentence):
        device = "cpu"
        prompt_sentence = Params.prompt_templates[0].replace("[X]", sentence)
        sentence_num = len(self.tokenizer.tokenize(sentence))
        template_sentence = Params.prompt_templates[0].replace("[X]", "[X]"*sentence_num)
        prompt_encodings = self.tokenizer(prompt_sentence,
                                          padding=True,
                                          truncation=True,
                                          max_length=Params.max_length,
                                          return_tensors='pt')
        template_encodings = self.tokenizer(template_sentence,
                                            padding=True,
                                            truncation=True,
                                            max_length=Params.max_length,
                                            return_tensors='pt')
        sentence_embedding = self.model(prompt_input_ids=prompt_encodings['input_ids'].to(device),
                                        prompt_attention_mask=prompt_encodings['attention_mask'].to(device), 
                                        prompt_token_type_ids=prompt_encodings['token_type_ids'].to(device), 
                                        template_input_ids=template_encodings['input_ids'].to(device), 
                                        template_attention_mask=template_encodings['attention_mask'].to(device), 
                                        template_token_type_ids=template_encodings['token_type_ids'].to(device))
        return sentence_embedding
    
    
    def calculate_sentence_similarity(self, sentence1, sentence2):
        # sentence1_embedding = self.calculate_sentence_mask_embedding(sentence1)
        # sentence2_embedding = self.calculate_sentence_mask_embedding(sentence2)
        sentence1_embedding = self.calculate_sentence_embedding(sentence1)
        sentence2_embedding = self.calculate_sentence_embedding(sentence2)
        similarity = torch.cosine_similarity(sentence1_embedding, sentence2_embedding, dim=-1)
        similarity = float(similarity.item())
        return similarity
    
    
class SBERTRetrieval(object):
    def __init__(self, pretrained_model_path, sbert_path, pool_type, dropout):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        model = SBERT(Params.pretrained_model, pool_type, dropout)
        self.checkpoint = torch.load(sbert_path, map_location='cpu')
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        self.model = model
        
        
    def print_checkpoint_info(self):
        loss = self.checkpoint['train_loss']
        epoch = self.checkpoint['epoch']
        model_info = {'loss': loss, 'epoch': epoch}
        return model_info
        
        
    def calculate_sentence_embedding(self, sentence):
        device = "cpu"
        input_encodings = self.tokenizer(sentence,
                                         padding=True,
                                         truncation=True,
                                         max_length=Params.max_length,
                                         return_tensors='pt')
        sentence_embedding = self.model(input_encodings['input_ids'].to(device),
                                        input_encodings['attention_mask'].to(device),
                                        input_encodings['token_type_ids'].to(device))
        return sentence_embedding
    
    
    def calculate_sentence_similarity(self, sentence1, sentence2):
        sentence1 = sentence1.strip()
        sentence2 = sentence2.strip()
        sentence1_embedding = self.calculate_sentence_embedding(sentence1)
        sentence2_embedding = self.calculate_sentence_embedding(sentence2)
        similarity = torch.cosine_similarity(sentence1_embedding, sentence2_embedding, dim=-1)
        similarity = float(similarity.item())
        return similarity
    

class CoSentRetrieval(object):
    def __init__(self, pretrained_model_path, cosent_path):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        model = CoSent(Params.pretrained_model, Params.cosent_pool_type, Params.cosent_dropout)
        self.checkpoint = torch.load(cosent_path, map_location='cpu')
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        self.model = model
        
        
    def print_checkpoint_info(self):
        loss = self.checkpoint['loss']
        epoch = self.checkpoint['epoch']
        model_info = {'loss': loss, 'epoch': epoch}
        return model_info
        
        
    def calculate_sentence_embedding(self, sentence):
        device = "cpu"
        input_encodings = self.tokenizer(sentence,
                                         padding=True,
                                         truncation=True,
                                         max_length=Params.max_length,
                                         return_tensors='pt')
        sentence_embedding = self.model(input_encodings['input_ids'].to(device),
                                        input_encodings['attention_mask'].to(device),
                                        input_encodings['token_type_ids'].to(device))
        return sentence_embedding
    
    
    def calculate_sentence_similarity(self, sentence1, sentence2):
        sentence1 = sentence1.strip()
        sentence2 = sentence2.strip()
        sentence1_embedding = self.calculate_sentence_embedding(sentence1)
        sentence2_embedding = self.calculate_sentence_embedding(sentence2)
        similarity = torch.cosine_similarity(sentence1_embedding, sentence2_embedding, dim=-1)
        similarity = float(similarity.item())
        return similarity
    

simcse_retrieval = SimCSERetrieval(Params.pretrained_model, Params.simcse_model, Params.pool_type, Params.simcse_dropout)
logger.info("start simcse model succussfully!")
esimcse_repeat_retrieval = ESimCSERetrieval(Params.pretrained_model, Params.esimcse_repeat_model, Params.esimcse_repeat_dropout)
logger.info("start esimcse repeat model succussfully!")
esimcse_same_retrieval = ESimCSERetrieval(Params.pretrained_model, Params.esimcse_same_model, Params.esimcse_same_dropout)
logger.info("start esimcse same model succussfully!")
esimcse_multi_retrieval = ESimCSERetrieval(Params.pretrained_model, Params.esimcse_multi_model, Params.esimcse_multi_dropout)
logger.info("start esimcse multi model succussfully!")
promptbert_retrieval = PromptBertRetrieval(Params.pretrained_model, Params.promptbert_model, Params.promptbert_dropout)
logger.info("start promptbert model succussfully!")
sbert_retrieval = SBERTRetrieval(Params.pretrained_model, Params.sbert_model, Params.sbert_pool_type, Params.sbert_dropout)
logger.info("start sbert model succussfully!")
cosent_retrieval = CoSentRetrieval(Params.pretrained_model, Params.cosent_model)
logger.info("start cosent model succussfully!")


if __name__ == "__main__":
    # model_path = "models/esimcse_0.32_0.15_160.pth"
    # model_path = "models/esimcse_multi_0.15_64.pth"
    # model_path = "models/esimcse_0.15_64.pth"
    
    
    
    # simcse_retrieval = SimCSERetrieval(Params.pretrained_model, Params.simcse_model, Params.pool_type, Params.simcse_dropout)
    # model_info = simcse_retrieval.print_checkpoint_info()
    # print(model_info)
    
    model_info = sbert_retrieval.print_checkpoint_info()
    print(model_info)
    
    while True:
        print("input your sentence1:")
        sentence1 = input()
        print("input your sentence2:")
        sentence2 = input()
        
        sbert_sentence_similarity = sbert_retrieval.calculate_sentence_similarity(sentence1, sentence2)
        # promptbert_sentence_similarity = prom.calculate_sentence_similarity(sentence1, sentence2)
        # print("simcse sim: {}, promptbert sim: {}".format(simcse_sentence_similarity, promptbert_sentence_similarity))
        print("sbert similarity: {}".format(sbert_sentence_similarity))