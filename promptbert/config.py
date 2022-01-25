

class Params:
    corpus_file = "data/questions.txt"
    # corpus_file = "data/examples.txt"
    epoches = 100
    batch_size = 32
    max_length = 64
    dropout = 0.15
    learning_rate = 3e-5
    tao = 0.05
    display_interval = 1000
    replace_token = "[X]"
    mask_token = "[MASK]"
    pretrained_model_path = "hfl/chinese-roberta-wwm-ext-large"
    prompt_templates = ['"{}"，它的意思是[MASK]。'.format(replace_token), '"{}"，这句话的意思是[MASK]。'.format(replace_token)]
    promptbert_path = "./models/promptbert_1231.pth"
    topk_sim_question_file = "results/finder_sim_clusters.txt"
    dialogues_file = "data/training_dialogues.txt"
    dialogues_file = "data/examples.txt"
    
    sim_topk_num = 100
    sim_threshold = 0.75
    num_threshold = 15
    nlist = 64