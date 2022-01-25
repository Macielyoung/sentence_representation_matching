

class Params:
    epoches = 100
    batch_size = 32
    max_length = 32
    learning_rate = 2e-5
    dropout = 0.2
    warmup_steps = 100
    display_interval = 500
    pretrained_model = "hfl/chinese-roberta-wwm-ext-large"
    sbert_model = "models/sbert_0106.pth"
    # pretrained_model = "clue/roberta_chinese_pair_large"
    pool_type = "mean"
    train_file = "data/train_dataset.csv"
    test_file = "data/test_dataset.csv"