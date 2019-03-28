import pandas as pd
import os
os.chdir('/Users/liyuan/desktop/CSAir/codes')
import re
import csv
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import jieba
import jieba.posseg as pseg
import jieba.analyse
import glob
import codecs
import random

from tokenization import Tokenization

class unlabeled_preprocess(object):
    def __init__(self):
        self.fk_reviews = []
    
    def load_unlabeled_data(self):
        # load data: 2018年季度APP反馈数据分析/反馈数据2018年6-7-8月数据
        F = codecs.open('../Source_Data/第二批data/2018年季度APP反馈数据分析/反馈数据2018年6-7-8月数据.csv','r','utf-8')
        input_data =F.readlines()[2:]
        F.close()

        # get list of reviews
        fk_review_list_678 = []
        for i in range(len(input_data)):
            try:
                review = input_data[i].split(',')[9]
                fk_review_list_678.append(review)
            except:
                continue

        # load data: 2018年季度APP反馈数据分析/反馈数据3-5月原始数据.csv
        F = codecs.open('../Source_Data/第二批data/2018年季度APP反馈数据分析/反馈数据3-5月原始数据.csv','r','utf-8')
        input_data =F.readlines()[1:]
        F.close()

        # get list of reviews
        fk_review_list_345 = []
        for i in range(len(input_data)):
            try:
                review = input_data[i].split(',')[7]
                fk_review_list_345.append(review)
            except:
                continue

        # load data: 2018年季度APP反馈数据分析/反馈数据3-5月原始数据.csv
        F = codecs.open('../Source_Data/第二批data/2018年季度APP反馈数据分析/反馈数据9-10-11.csv','r','utf-8')
        input_data =F.readlines()[2:]
        F.close()

        # get list of reviews
        fk_review_list_91011 = []
        for i in range(len(input_data)):
            try:
                review = input_data[i].split(',')[12]
                fk_review_list_91011.append(review)
            except:
                continue

        # concat unlabeled reviews into one file
        self.fk_reviews += fk_review_list_345
        self.fk_reviews += fk_review_list_678
        self.fk_reviews += fk_review_list_91011
        print('反馈数据 in total has %d reviews'% len(self.fk_reviews))
        print('反馈数据举3例：', self.fk_reviews[:3])
        return self.fk_reviews

    def sample_unlabeled(self,sample_size, output_file_path):
        # sample 5000 user reviews from fk_reivews; 
        # need to keep the index of each sampled reviews; ultimately would need manual evualuation
        # get the index list of the sampled reviews
        sample_index_list = [i for i in random.sample(range(len(self.fk_reviews)),sample_size)]
        sample_reviews = [self.fk_reviews[i] for i in random.sample(range(len(self.fk_reviews)), sample_size)]
        
        with open(output_file_path,'w',newline='') as output_file:
            for line in sample_reviews:
                output_file.write(line + '\n') 
        return sample_reviews

    