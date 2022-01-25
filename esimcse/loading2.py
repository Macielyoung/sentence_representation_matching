from transformers import AutoTokenizer
import pandas as pd
import re
import random
from config import Params


class DataLoading:
    def __init__(self, question_file, qa_file, pretrained_model):
        self.question_file = question_file
        self.qa_file = qa_file
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        
        
    def load_chat(self):
        question_df = pd.read_csv(self.question_file)
        # print(question_df.shape)
        question_df = question_df.fillna("")
        question_df['length'] = question_df.apply(lambda x: len(x['content']), axis=1)
        question_df['content'] = question_df.apply(lambda x: self.preprocess_sentence(x['content']), axis=1)
        question_df = question_df[(question_df.content != "") & (question_df.length < 32)]
        questions = list(question_df.content.unique())
        return questions
    
    
    def read_qa(self):
        qa_df = pd.read_csv(self.qa_file)
        qa_questions = list(qa_df['question'])
        qa_questions = [self.preprocess_sentence(item) for item in qa_questions]
        return qa_questions
    
    
    def load_data(self):
        chat_questions = self.load_chat()
        qa_questions = self.read_qa()
        questions = chat_questions + qa_questions
        return questions
        
        
    def preprocess_sentence(self, sentence):
        text = re.sub("\[(.*?)\]", "", sentence)
        text = re.sub("\【(.*?)\】", "", text)
        text = text.replace("\n", "")
        text = re.sub("\/:[a-zA-Z@)(]*", "", text)
        new_text = re.sub("[0-9]*", "", text)
        if new_text == "":
            return new_text
        return text
    
    
    def repeat_word(self, sentence):
        '''
        @function: 重复句子中的部分token
        
        @input:
        sentence: string，输入语句
        
        @return:
        dup_sentence: string，重复token后生成的句子
        '''
        word_tokens = self.tokenizer.tokenize(sentence)
        
        # dup_len ∈ [0, max(2, int(dup_rate ∗ N))]
        max_len = max(2, int(Params.dup_rate * len(word_tokens)))
        # 防止随机挑选的数值大于token数量
        dup_len = min(random.choice(range(max_len+1)), len(word_tokens))
        
        random_indices = random.sample(range(len(word_tokens)), dup_len)
        # print(max_len, dup_len, random_indices)
        
        dup_word_tokens = []
        for index, word in enumerate(word_tokens):
            dup_word_tokens.append(word)
            if index in random_indices:
                dup_word_tokens.append(word)
        dup_sentence = "".join(dup_word_tokens)
        return dup_sentence
        
    
    def generate_pos_dataset(self, questions):
        # 使用重复单词的方法生成相似样本对正例
        pos_question_pairs = []
        for qid, question in enumerate(questions):
            pos_question = self.repeat_word(question)
            pos_question_pairs.append([question, pos_question])
            if qid % 10000 == 0:
                print("qid: {}, question: {}, pos_question: {}".format(qid, question, pos_question), flush=True)
        return pos_question_pairs
    
    
    def generate_pos_dataset2(self, questions):
        # 使用同样的句子作为正样本对
        pos_question_pairs = []
        for qid, question in enumerate(questions):
            # pos_question = self.repeat_word(question)
            pos_question_pairs.append([question, question])
            if qid % 10000 == 0:
                print("qid: {}, question: {}, pos_question: {}".format(qid, question, question), flush=True)
        return pos_question_pairs
    
    
    def save_question_pairs(self, pos_question_pairs, pair_path):
        question_pairs = [{'question': pair[0], 'pos_question': pair[1]} for pair in pos_question_pairs]
        question_df = pd.DataFrame(question_pairs)
        question_df.to_csv(pair_path)
    
    
    def save_questions(self, questions, question_path):
        questions = [item+"\n" for item in questions]
        with open(question_path, 'w') as f:
            f.writelines(questions)
    
    
if __name__ == "__main__":
    data_loading = DataLoading(Params.question_file, Params.qa_file, Params.pretrained_model)
    # questions = data_loading.load_data()
    # print(len(questions))
    # print(questions[:10])
    
    # question = "666你好"
    # text = data_loading.preprocess_sentence(question)
    # print(text)
    
    # questions = data_loading.load_data()
    # print("question num: {}".format(len(questions)))
    # question_path = "data/questions.txt"
    # data_loading.save_questions(questions, question_path)
    
    # sentence = "担心交钱以后，没有后续服务了"
    # sim_sentence = data_loading.get_synonym_sentence(sentence)
    # print(sim_sentence)
    
    questions = data_loading.load_data()
    print("question num: {}".format(len(questions)))
    
    # pos_question_pairs = data_loading.generate_pos_dataset3(questions)
    # print("question pair num: {}".format(len(pos_question_pairs)))
    
    # data_loading.save_question_pairs(pos_question_pairs, Params.pair_file)
    # print("save pos pair file done")