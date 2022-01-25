import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import Params


class LoadDataset():
    def __init__(self, data_file):
        data_df = pd.read_csv(data_file)
        self.data_df = data_df.fillna("")
        
    
    def get_dataset(self):
        question1_list = list(self.data_df['question1'])
        question2_list = list(self.data_df['question2'])
        label_list = list(self.data_df['label'])
        return question1_list, question2_list, label_list
    
    
    def get_encodings(self, tokenzier, questions):
        question_encodings = tokenzier(questions,
                                       truncation=True,
                                       padding=True,
                                       max_length=Params.max_length,
                                       return_tensors='pt')
        return question_encodings


class PairDataset(Dataset):
    def __init__(self, q1_encodings, q2_encodings, labels):
        self.q1_encodings = q1_encodings
        self.q2_encodings = q2_encodings
        self.labels = labels
    
    
    # 读取单个样本
    def __getitem__(self, idx):
        item1 = {key: torch.tensor(val[idx]) for key, val in self.q1_encodings.items()}
        labels = torch.tensor(int(self.labels[idx]))
        item2 = {key: torch.tensor(val[idx]) for key, val in self.q2_encodings.items()}
        return item1, item2, labels
    
    
    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    print(1)