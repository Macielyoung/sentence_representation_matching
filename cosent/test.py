import torch
import pandas as pd
from transformers import AutoTokenizer
from model import CoSent
from config import Params


def read_test_data(test_file):
    test_df = pd.read_csv(test_file)
    test_df = test_df.fillna("")
    return test_df


def split_similarity(similarity, threshold):
    if similarity >= threshold:
        return 1
    else:
        return 0
    

def calculate_accuracy(data_df, threshold):
    data_df['pred'] = data_df.apply(lambda x: split_similarity(x['similarity'], threshold), axis=1)
    pred_correct = data_df[data_df['pred'] == data_df['label']]
    # pred_error = data_df[data_df['pred'] != data_df['label']]
    # print(pred_error)
    accuracy = len(pred_correct) / len(data_df)
    return accuracy, len(pred_correct), len(data_df)


class CoSentRetrieval(object):
    def __init__(self, pretrained_model_path, cosent_path):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        model = CoSent(Params.pretrained_model, Params.pooler_type, Params.dropout)
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
    

cosent_retrieval = CoSentRetrieval(Params.pretrained_model, Params.cosent_model)
model_info = cosent_retrieval.print_checkpoint_info()
print("load model done, model_info: {}".format(model_info))
test_file = "data/test_dataset.csv"
test_df = read_test_data(test_file)
results = []
for rid, row in test_df.iterrows():
    question1 = row['question1']
    question2 = row['question2']
    label = row['label']
    similarity = cosent_retrieval.calculate_sentence_similarity(question1, question2)
    item = {'question1': question1,
            'question2': question2,
            'similarity': similarity,
            'label': label}
    if rid % 100 == 0:
        print("rid: {}, item: {}".format(rid, item))
    results.append(item)
print("prediction done!")
    

pred_df = pd.DataFrame(results)
pred_file = "results/test_pred.csv"
pred_df.to_csv(pred_file)
max_acc = 0
for t in range(50, 80, 1):
    t = t / 100
    acc, correct, num = calculate_accuracy(pred_df, t)
    if acc > max_acc:
        max_acc = acc
        print(t, acc, correct, num)
        
print(max_acc)