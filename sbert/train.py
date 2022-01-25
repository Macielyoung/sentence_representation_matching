import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from loading import LoadDataset, PairDataset
from model import SBERT
from config import Params
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def compute_loss(similarity, label, loss_fn):
    # mse loss
    loss = loss_fn(similarity, label)
    return loss


def compute_acc(similarity, label):
    pred = (similarity >= 0.5).long()
    acc = accuracy_score(pred.detach().cpu().numpy(), label.cpu().numpy())
    return acc


def testing(model, test_loader):
    model.eval()
    test_loss, test_acc = [], []
    for test_q1, test_q2, test_label in test_loader:
        test_q1_input_ids = test_q1['input_ids'].to(device)
        test_q1_attention_mask = test_q1['attention_mask'].to(device)
        test_q1_token_type_ids = test_q1['token_type_ids'].to(device)
        
        test_q2_input_ids = test_q2['input_ids'].to(device)
        test_q2_attention_mask = test_q2['attention_mask'].to(device)
        test_q2_token_type_ids = test_q2['token_type_ids'].to(device)
        
        test_label = test_label.float().to(device)
            
        test_q1_embedding = model(test_q1_input_ids, test_q1_attention_mask, test_q1_token_type_ids)
        test_q2_embedding = model(test_q2_input_ids, test_q2_attention_mask, test_q2_token_type_ids)
        test_similarity = torch.cosine_similarity(test_q1_embedding, test_q2_embedding, dim=1)
        batch_test_loss = compute_loss(test_similarity, test_label, loss_fn)
        batch_test_acc = compute_acc(test_similarity, test_label)
        
        test_loss.append(batch_test_loss.item())
        test_acc.append(batch_test_acc)
    
    test_avg_loss = np.mean(test_loss)
    test_avg_acc = np.mean(test_acc)
    return test_avg_loss, test_avg_acc
    

train_loading = LoadDataset(Params.train_file)
test_loading = LoadDataset(Params.test_file)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(Params.pretrained_model)

# load train dataset
train_question1, train_question2, train_labels = train_loading.get_dataset()
train_q1_encodings = train_loading.get_encodings(tokenizer, train_question1)
train_q2_encodings = train_loading.get_encodings(tokenizer, train_question2)
train_dataset = PairDataset(train_q1_encodings, train_q2_encodings, train_labels)
train_loader = DataLoader(train_dataset, 
                          batch_size=Params.batch_size, 
                          shuffle=True)

# load test dataset
test_question1, test_question2, test_labels = test_loading.get_dataset()
test_q1_encodings = test_loading.get_encodings(tokenizer, test_question1)
test_q2_encodings = test_loading.get_encodings(tokenizer, test_question2)
test_dataset = PairDataset(test_q1_encodings, test_q2_encodings, test_labels)
test_loader = DataLoader(test_dataset,
                         batch_size=Params.batch_size)

# load model
model = SBERT(Params.pretrained_model, Params.pool_type, Params.dropout)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# load loss function
loss_fn = nn.MSELoss()

# load optimizer
optim = AdamW(model.parameters(), lr=Params.learning_rate)
total_steps = len(train_loader)
scheduler = get_linear_schedule_with_warmup(optim, 
                                            num_warmup_steps = Params.warmup_steps, # Default value in run_glue.py
                                            num_training_steps = total_steps)


best_loss = 100000
for epoch in range(Params.epoches):
    model.train()
    batch_num = 0
    epoch_losses = []
    epoch_acces = []
    for q1, q2, label in tqdm(train_loader):
        
        q1_input_ids = q1['input_ids'].to(device)
        q1_attention_mask = q1['attention_mask'].to(device)
        q1_token_type_ids = q1['token_type_ids'].to(device)
        
        q2_input_ids = q2['input_ids'].to(device)
        q2_attention_mask = q2['attention_mask'].to(device)
        q2_token_type_ids = q2['token_type_ids'].to(device)
        
        label = label.float().to(device)
        # print(q1_input_ids, q2_input_ids, label)
        
        optim.zero_grad()
        q1_embedding = model(q1_input_ids, q1_attention_mask, q1_token_type_ids)
        q2_embedding = model(q2_input_ids, q2_attention_mask, q2_token_type_ids)
        similarity = torch.cosine_similarity(q1_embedding, q2_embedding, dim=1)
        batch_loss = compute_loss(similarity, label, loss_fn)
        # print("batch loss: {}, type: {}".format(batch_loss.item(), batch_loss.dtype))
        batch_acc = compute_acc(similarity, label)
        
        # 梯度更新+裁剪
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 参数更新
        optim.step()
        scheduler.step()
        
        epoch_losses.append(batch_loss.item())
        epoch_acces.append(batch_acc)
        
        if batch_num % Params.display_interval == 0:
            print("Epoch: {}, batch: {}/{}, loss: {}, acc: {}".format(epoch, batch_num, total_steps, batch_loss, batch_acc), flush=True)
        batch_num += 1
        
    epoch_avg_loss = np.mean(epoch_losses)
    epoch_avg_acc = np.mean(epoch_acces)
    print("Epoch: {}, avg loss: {}, acc: {}".format(epoch, epoch_avg_loss, epoch_avg_acc), flush=True)
    if epoch_avg_loss < best_loss:
        test_avg_loss, test_avg_acc = testing(model, test_loader)
        
        print("Epoch: {}, train loss: {}, acc: {}, test loss: {}, acc: {}, save best model".format(epoch, epoch_avg_loss, epoch_avg_acc, test_avg_loss, test_avg_acc), flush=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'train_loss': epoch_avg_loss,
            'train_acc': epoch_avg_acc,
            'test_loss': test_avg_loss,
            'test_acc': test_avg_acc
        }, Params.sbert_model)
        best_loss = epoch_avg_loss
        
        
        