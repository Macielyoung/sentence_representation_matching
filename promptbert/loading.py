# -*- coding: utf-8 -*-
# @Time    : 2022/9/17
# @Author  : Maciel

from torch.utils.data import DataLoader


class MatchingDataSet:    
    def read_train_file(self, trainfile, devfile, testfile, filetype):
        sents = []
        if filetype == "txt":
            with open(trainfile, 'r') as f:
                for line in f.readlines():
                    _, s1, s2, _ = line.strip().split(u"||")
                    sents.append(s1)
                    sents.append(s2)
            with open(devfile, 'r') as f:
                for line in f.readlines():
                    _, s1, s2, _ = line.strip().split(u"||")
                    sents.append(s1)
                    sents.append(s2)
            with open(testfile, 'r') as f:
                for line in f.readlines():
                    _, s1, s2, _ = line.strip().split(u"||")
                    sents.append(s1)
                    sents.append(s2)
        return sents
    
    def read_eval_file(self, file, filetype):
        sents = []
        if filetype == "txt":
            with open(file, 'r') as f:
                for line in f.readlines():
                    _, s1, s2, s = line.strip().split(u"||")
                    item = {'sent1': s1,
                            'sent2': s2,
                            'score': float(s)}
                    sents.append(item)
        return sents
    
    
if __name__ == "__main__":
    trainfile = "../dataset/STS-B/train.txt"
    devfile = "../dataset/STS-B/dev.txt"
    testfile = "../dataset/STS-B/test.txt"
    match_dataset = MatchingDataSet()
    
    train_list = match_dataset.read_train_file(trainfile, devfile, testfile, "txt")
    print(train_list[:5])
    
    train_lengths = [len(sentence) for sentence in train_list]
    max_len = max(train_lengths)
    
    
    dev_list = match_dataset.read_eval_file(devfile, "txt")
    dev_sen1_length = [len(d['sent1']) for d in dev_list]
    dev_sen2_length = [len(d['sent2']) for d in dev_list]
    max_sen1 = max(dev_sen1_length)
    max_sen2 = max(dev_sen2_length)
    print(max_len, max_sen1, max_sen2)
    # dev_loader = DataLoader(dev_list,
    #                         batch_size=8)
    # for batch in dev_loader:
    #     print(batch)
    #     exit(0)