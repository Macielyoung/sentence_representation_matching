from flask import Flask, request, jsonify
from retrieval import simcse_retrieval, esimcse_same_retrieval, esimcse_repeat_retrieval, esimcse_multi_retrieval, promptbert_retrieval
from retrieval import sbert_retrieval, cosent_retrieval


app = Flask(__name__)
@app.route('/compare', methods=['GET', 'POST'])
def compare():
    sentence1 = request.args.get("sentence1", "")
    sentence2 = request.args.get("sentence2", "")
    channel = request.args.get("channel", "")
    sentence1 = strip_sentence(sentence1)
    sentence2 = strip_sentence(sentence2)
    
    if channel == "0":
        model_name = "simcse"
        model_desc = "use simcse model for data augmentation and compare positive samples"
        model_info = simcse_retrieval.print_checkpoint_info()
        model_info['model_name'] = model_name
        model_info['model_desc'] = model_desc
        similarity = simcse_retrieval.calculate_sentence_similarity(sentence1, sentence2)
    elif channel == "1":
        model_name = "esimcse_same_positive"
        model_desc = "use same sentence as positive pairs and construct a negative queue"
        model_info = esimcse_same_retrieval.print_checkpoint_info()
        model_info['model_name'] = model_name
        model_info['model_desc'] = model_desc
        similarity = esimcse_same_retrieval.calculate_sentence_similarity(sentence1, sentence2)
    elif channel == "2":
        model_name = "esimcse_repeat_positive"
        model_desc = "use repeat word for data augmentation as positive pairs and construct a negative queue"
        model_info = esimcse_repeat_retrieval.print_checkpoint_info()
        model_info['model_name'] = model_name
        model_info['model_desc'] = model_desc
        similarity = esimcse_repeat_retrieval.calculate_sentence_similarity(sentence1, sentence2)
    elif channel == "3":
        model_name = "esimcse_multi_positive"
        model_desc = "Multi task loss: use same sentence and repeat word as positive pairs and construct a negative queue"
        model_info = esimcse_multi_retrieval.print_checkpoint_info()
        model_info['model_name'] = model_name
        model_info['model_desc'] = model_desc
        similarity = esimcse_multi_retrieval.calculate_sentence_similarity(sentence1, sentence2)
    elif channel == "4":
        model_name = "promptbert"
        model_desc = "use different templates to generate sentence embedding as positive pairs"
        model_info = promptbert_retrieval.print_checkpoint_info()
        model_info['model_name'] = model_name
        model_info['model_desc'] = model_desc
        similarity = promptbert_retrieval.calculate_sentence_similarity(sentence1, sentence2)
    elif channel == "5":
        model_name = "sbert"
        model_desc = "train sentence-bert structure model with cosine similarity and mse loss, using high prediction probability cases as training dataset"
        model_info = sbert_retrieval.print_checkpoint_info()
        model_info['model_name'] = model_name
        model_info['model_desc'] = model_desc
        similarity = sbert_retrieval.calculate_sentence_similarity(sentence1, sentence2)
    elif channel == "6":
        model_name = "cosent"
        model_desc = "train cosent structure model with contrastive loss"
        model_info = cosent_retrieval.print_checkpoint_info()
        model_info['model_name'] = model_name
        model_info['model_desc'] = model_desc
        similarity = cosent_retrieval.calculate_sentence_similarity(sentence1, sentence2)
    else:
        model_info = {'model_name': 'your channel is illegal'}
        similarity = None
    
    sent_info = {'sentence1': sentence1, 'sentence2': sentence2, 'similarity': similarity}
    result = {'model_info': model_info, 'sentence_info': sent_info}
    data = {'code': 200, 'message': 'OK', 'data': result}
    resp = jsonify(data)
    return resp


def strip_sentence(sentence):
    sentence = sentence.strip().lower()
    sentence = sentence.replace("?", "").replace("？", "")
    return sentence


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)