import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from config import Params


class Loader():
    def __init__(self, data_file):
        data_df = pd.read_csv(data_file)
        self.data_df = data_df.fillna("")
        
    
    def get_dataset(self):
        sentences = []
        labels = []
        for _, row in self.data_df.iterrows():
            question1 = row['question1']
            question2 = row['question2']
            label = row['label']
            sentences.extend([question1, question2])
            labels.extend([label, label])
            # rows.append([question1, label])
            # rows.append([question2, label])
        return sentences, labels
    
    
class CustomerDataset(Dataset):
    def __init__(self, sentence, label, tokenizer):
        self.sentence = sentence
        self.label = label
        self.tokenizer = tokenizer
        
    
    def __len__(self):
        return len(self.sentence)
    
    
    def __getitem__(self, index):
        input_encodings = self.tokenizer(self.sentence[index])
        input_ids = input_encodings['input_ids']
        attention_mask = input_encodings['attention_mask']
        token_type_ids = input_encodings['token_type_ids']
        item = {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'label': self.label[index]}
        return item
    
    
def pad_to_maxlen(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids


def collate_fn(batch):
    # 按batch进行padding获取当前batch中最大长度
    max_len = max([len(d['input_ids']) for d in batch])
    
    # 如果当前最长长度超过设定的全局最大长度，则取全局最大长度
    max_len = max_len if max_len <= Params.max_length else Params.max_length
    
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []

    for item in batch:
        input_ids.append(pad_to_maxlen(item['input_ids'], max_len=max_len))
        attention_mask.append(pad_to_maxlen(item['attention_mask'], max_len=max_len))
        token_type_ids.append(pad_to_maxlen(item['token_type_ids'], max_len=max_len))
        labels.append(item['label'])
    
    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
    all_label_ids = torch.tensor(labels, dtype=torch.float)
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids

    
if __name__ == "__main__":
    data_file = "data/train_dataset.csv"
    loader = Loader(data_file)
    sentence, label = loader.get_dataset()
    print("load question file done!")
    
    tokenizer = AutoTokenizer.from_pretrained(Params.pretrained_model)
    dataset = CustomerDataset(sentence, label, tokenizer)
    # print(dataset[0])
    train_loader = DataLoader(dataset,
                              shuffle=False,
                              batch_size=Params.batch_size,
                              collate_fn=collate_fn)
    
    for dl in train_loader:
        print(dl)
        exit(0)