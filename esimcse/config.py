

class Params:
    dup_rate = 0.25
    dropout = 0.15
    batch_size = 64
    max_length = 32
    queue_num = 64
    # queue_num = 20
    lr = 1e-5
    epoches = 100
    tao = 0.05
    momentum = 0.995
    display_interval = 100
    sim_topk_num = 100
    sim_threshold = 0.75
    num_threshold = 15
    nlist = 64
    
    pretrained_model = "hfl/chinese-roberta-wwm-ext-large"
    esimcse_model_dir = "models/esimcse"
    esimcse_model = "models/esimcse_0.15_64.pth"
    question_file = "data/raw_questions.csv"
    dialogues_file = "data/questions.txt"
    qa_file = "data/qa.csv"
    pair_file = "data/question_pair.csv"
    topk_sim_question_file = "results/finder_sim_clusters.txt"