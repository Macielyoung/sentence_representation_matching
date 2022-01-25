import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from loading import Loader, CustomerDataset, collate_fn
from config import Params
from model import CoSent
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def compute_sim(y_pred):
    # 1. 对输出的句子向量进行l2归一化   后面只需要对应为相乘  就可以得到cos值了
    norms = (y_pred ** 2).sum(axis=1, keepdims=True) ** 0.5
    # y_pred = y_pred / torch.clip(norms, 1e-8, torch.inf)
    y_pred = y_pred / norms

    # 2. 奇偶向量相乘
    sim = torch.sum(y_pred[::2] * y_pred[1::2], dim=1)
    return sim


def compute_acc(y_true, y_sim, threshold):
    # 1. 取出真实的标签(每两行是一个文本匹配对)
    y_true = y_true[::2]    # tensor([1, 0, 1]) 真实的标签
    
    # 2. 根据阈值分割
    y_pred_label = (y_sim >= threshold).float()
    acc = accuracy_score(y_pred_label.detach().cpu().numpy(), y_true.cpu().numpy())
    return acc
    

def compute_loss(y_true, y_sim):
    # 1. 取出真实的标签(每两行是一个文本匹配对)
    y_true = y_true[::2]    # tensor([1, 0, 1]) 真实的标签

    # 2. 根据句子间相似度进行放缩
    y_sim = y_sim * 20

    # 3. 取出负例-正例的差值
    y_sim = y_sim[:, None] - y_sim[None, :]  # 这里是算出所有位置 两两之间余弦的差值
    # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
    y_true = y_true[:, None] < y_true[None, :]   # 取出负例-正例的差值
    y_true = y_true.float()
    y_sim = y_sim - (1 - y_true) * 1e12
    y_sim = y_sim.view(-1)
    if torch.cuda.is_available():
        y_sim = torch.cat((torch.tensor([0]).float().cuda(), y_sim), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    else:
        y_sim = torch.cat((torch.tensor([0]).float(), y_sim), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
        
    return torch.logsumexp(y_sim, dim=0)


data_file = "data/train_dataset.csv"
loader = Loader(data_file)
sentence, label = loader.get_dataset()
print("load question file done!")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(Params.pretrained_model)
train_dataset = CustomerDataset(sentence, label, tokenizer)
train_loader = DataLoader(train_dataset,
                          shuffle=False,
                          batch_size=Params.batch_size,
                          collate_fn=collate_fn)
print("tokenize all batch done!")

total_steps = len(train_loader) * Params.epoches
train_optimization_steps = int(len(train_dataset) / Params.batch_size) * Params.epoches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load model
model = CoSent(Params.pretrained_model, Params.pooler_type, Params.dropout)
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# load optimizer and scheduler
optimizer = AdamW(optimizer_grouped_parameters, lr=Params.learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
                                            num_training_steps=total_steps)

best_loss = 1000000
print("start training...")
for epoch in range(Params.epoches):
    model.train()
    epoch_losses = []
    epoch_acces = []
    step = 0
    
    for batch in tqdm(train_loader):
        input_ids, input_mask, segment_ids, label_ids = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        
        output = model(input_ids, input_mask, segment_ids)
        y_sim = compute_sim(output)
        loss = compute_loss(label_ids, y_sim)
        acc = compute_acc(label_ids, y_sim, Params.threshold)
        
        # update gradient and scheduler
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if step % Params.display_steps == 0:
            print("Epoch: {}, Step: {}, Batch loss: {}, acc: {}".format(epoch, step, loss.item(), acc), flush=True)
        epoch_losses.append(loss.item())
        epoch_acces.append(acc)
        step += 1
    
    avg_epoch_loss = np.mean(epoch_losses)
    avg_epoch_acc = np.mean(epoch_acces)
    print("Epoch: {}, avg loss: {}, acc: {}".format(epoch, avg_epoch_loss, avg_epoch_acc), flush=True)
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        print("Epoch: {}, best loss: {}, acc: {} save model".format(epoch, best_loss, avg_epoch_acc), flush=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': avg_epoch_loss,
            'acc': avg_epoch_acc,
        }, Params.cosent_model)