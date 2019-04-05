from tokenization import Tokenization
from prepare_data import PrepareData
from modeling import Modeling
import os
os.chdir('/Users/liyuan/desktop/CSAir/codes')

import numpy as np
import pandas as pd

class ReviewClassify(object):
    def __init__(self, model):
        self.model = model
        self.data = pd.DataFrame()
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()

        self.labels_index = {}
        self.class_priors = {}
        self.class_size = {}

        self.prob_scores = np.array([])
        self.threshold_dct = {}
        self.class_label_dct = {}
        self.class_reviews_dct = {}
    
    def load_data(self, input_data_path):
        data_p = PrepareData()
        self.data = data_p.load_data('../res/all_labeled_data.csv')
        self.train, self.test = data_p.split_data()
        X_train, y_train, X_test, y_test = data_p.preprocess_data()
        self.labels_index = data_p.get_labels_index()
        self.class_priors = data_p.get_class_priors()
        self.class_size = data_p.get_class_size()
        return X_train, y_train, X_test, y_test

    def classify_reviews(self, input_data_path):
        '''this function incorporates the previous functions together: fit classifier + manually set threshold + get predicted results '''
        # part1: load data
        data_p = PrepareData()
        self.data = data_p.load_data(input_data_path)
        self.train, self.test = data_p.split_data()
        X_train, y_train, X_test, y_test = data_p.preprocess_data()
        self.labels_index = data_p.get_labels_index()
        self.class_priors = data_p.get_class_priors()
        self.class_size = data_p.get_class_size()

        # part2: modeling
        # funcitons below should return different results when passing on different models
        m = Modeling(X_train, y_train, X_test, y_test)
        self.prob_scores = m.get_label_prob(self.model)
        self.threshold_dct = data_p.get_class_threshold(self.prob_scores)
        # generate label based on threshold of each class
        self.class_label_dct = m.gen_label_dct(self.prob_scores,self.threshold_dct)
        self.class_reviews_dct = m.map_label_to_review(self.class_label_dct)
        # checking
        # print('user review indices that are classified as high probability of class 0:', self.class_reviews_dct[0])
        return self.class_reviews_dct


