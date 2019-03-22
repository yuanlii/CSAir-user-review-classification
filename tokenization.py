import pandas as pd
import os
import re
import glob
import numpy as np
import time
import jieba
import jieba.posseg as pseg
import jieba.analyse
import warnings
warnings.filterwarnings('ignore')
os.chdir('/Users/liyuan/desktop/CSAir')

class Tokenization():
#     def __init__(self, input_data, output_name, stopwords):
    def __init__(self, input_path, output_name, stopwords):
        #self.input = input_data[:]
        self.input_path = input_path
        self.input = []
        self.output_name = str(output_name)
        self.sentences = []
        self.tfidf_score = []
        self.stopwords = stopwords
    
    def load_input_data(self):
        with open(self.input_path, 'r', encoding='utf-8') as input_file:
            self.input +=  input_file.readlines()
            self.input = self.input[:]
        return self.input
        
    def get_tokenized_sents(self):        
        for sent in self.input:
            tokenized_sent = ' '.join(word for word in jieba.cut(sent.strip()) if word not in self.stopwords)
            # remove digits
            tokenized_sent = re.sub(r'\d+','',tokenized_sent)
            # remove punctuation
            tokenized_sent = re.sub(r'[^\w\s]','', tokenized_sent)
            # remove non-chinese characters
            # match all Chinese words
            re_words = re.compile(u"[\u4e00-\u9fa5]+")
            res = re.findall(re_words, tokenized_sent)
            if res:
                valid_tokenized_sent = ' '.join([r for r in res])
            self.sentences.append(valid_tokenized_sent)
        
        with open(self.output_name + '.txt','w',newline='') as output_file:
            for line in self.sentences:
                output_file.write(line + '\n')  
 
        return self.sentences
    
    def get_topN_tf_idf(self, content, topK=20):
        tags = jieba.analyse.extract_tags(content, topK)
        return " ".join(tags)