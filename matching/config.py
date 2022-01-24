

class Params:
    esimcse_same_dropout = 0.15
    esimcse_repeat_dropout = 0.15
    esimcse_multi_dropout = 0.15
    promptbert_dropout = 0.1
    simcse_dropout = 0.3
    sbert_dropout = 0.2
    cosent_dropout = 0.15
    max_length = 32
    pool_type = "pooler"
    sbert_pool_type = "mean"
    cosent_pool_type = "mean"
    mask_token = "[MASK]"
    replace_token = "[X]"
    # prompt_templates = ['“[UNK]”，它的意思是[MASK]。', '“[UNK]”，这句话的意思是[MASK]。']
    prompt_templates = ['"{}"，它的意思是[MASK]。'.format(replace_token), '"{}"，这句话的意思是[MASK]。'.format(replace_token)]
    pretrained_model = "hfl/chinese-roberta-wwm-ext-large"
    esimcse_repeat_model = "models/esimcse_0.32_0.15_160.pth"
    esimcse_same_model = "models/esimcse_0.15_64.pth"
    esimcse_multi_model = "models/esimcse_multi_0.15_64.pth"
    promptbert_model = "models/promptbert_1231.pth"
    simcse_model = "models/simcse_1226.pth"
    sbert_model = "models/sbert_0106.pth"
    cosent_model = "models/cosent_0119.pth"