
        
class Params:
    delta_days = 16
    content_num = 10
    fromuser_num = 5
    sim_topk_num = 100
    sim_threshold = 0.75
    num_threshold = 15
    nlist = 64
    
    platform = "dev"
    question_file = "data/questions.csv"
    dialogues_file = "data/training_dialogues.txt"
    pretrained_model_path = "hfl/chinese-roberta-wwm-ext-large"
    simcse_model_path = "model/best_model.pth"
    topk_question_file = "results/topk_duplicate_questions"
    topk_sim_question_file = "results/finder_sim_clusters.txt"