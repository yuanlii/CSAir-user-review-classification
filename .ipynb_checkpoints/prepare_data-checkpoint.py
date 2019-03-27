import pandas as pd
import os
import re
import glob
import numpy as np
import time
import jieba
import jieba.posseg as pseg
import jieba.analyse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 

class PrepareData():
    def __init__(self):
        self.data = pd.DataFrame()
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        return self.data
    
    def split_data(self):
        self.train, self.test = train_test_split(self.data, test_size = 0.33, random_state=42)
        print('training data has %d examples' %len(self.train))
        print('test data has %d examples' %len(self.test))
        return self.train, self.test
    
    def preprocess_data(self):
        '''use countvectorizer and tf-idf transformer to get valid one-hot encoding for reviews'''
        # use countVectorizer for one-hot encoding
        count_v0= CountVectorizer();  
        counts_all = count_v0.fit_transform(self.data['review_tokens'])
        count_v1= CountVectorizer(vocabulary=count_v0.vocabulary_)  
        counts_train = count_v1.fit_transform(self.train.review_tokens)
        print ("the shape of train word vectors is "+repr(counts_train.shape))

        count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_)
        counts_test = count_v2.fit_transform(self.test.review_tokens)
        print ("the shape of test word vectors is "+repr(counts_test.shape))

        # implement tf-idf
        tfidftransformer = TfidfTransformer()
        train_data = tfidftransformer.fit(counts_train).transform(counts_train)
        test_data = tfidftransformer.fit(counts_test).transform(counts_test)
        
        X_train = train_data
        y_train = self.train.label_encoded
        X_test = test_data
        y_test = self.test.label_encoded
        return X_train, y_train, X_test, y_test
    
    def get_precision(self,y_pred, y_test):
        '''this function returns a precision score for the model'''
        num = 0
        y_pred = y_pred.tolist()
        for i,pred in enumerate(y_pred):
            if int(pred) == int(y_test.values[i]):
                num += 1
        precision = float(num) / len(y_pred)
        print('precision: '+'{:.2f}'.format(precision))
        return precision
   