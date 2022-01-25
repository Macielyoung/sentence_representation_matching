

class Params:
    epoches = 50
    max_length = 32
    batch_size = 64
    dropout = 0.15
    learning_rate = 3e-5
    threshold = 0.5
    gradient_accumulation_steps = 100
    display_steps = 500
    pooler_type = "mean"
    pretrained_model = "hfl/chinese-roberta-wwm-ext-large"
    cosent_model = "models/cosent.pth"
    