from tokenization import Tokenization
from prepare_data import PrepareData
from modeling import Modeling
import os
os.chdir('/Users/liyuan/desktop/CSAir/codes')

import numpy as np
import pandas as pd
from collections import defaultdict

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
        self.review_label = defaultdict(list)
        
        self.joined_test_data = pd.Series()

    
    def load_data(self, input_data_path):
        data_p = PrepareData()
        self.data = data_p.load_data(input_data_path)
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

    def reformat_review_label(self, class_reviews_dct):
        ''' re-organize predicted labels into ordered user reviews,e.g., {review0: [1], review1: [3,15], review2: [4,10,55], ...}; keys are the index of each review in test set (561 examples in total, starting from 0,1,2,..etc.)'''
        for i in range(10):
            for review in class_reviews_dct[i]:
                self.review_label[review].append(i)
        print('there are only %d user reviews picked by classes after manual setting threshold' %len(self.review_label))
        reviews_picked = self.review_label.keys() 
        # print('indices of user reviews in test data that are picked after manual setting threshold:', reviews_picked)
        return self.review_label
    
    def add_pred_to_data(self):
        # create a df for predicted labels
        d = {}
        d['index'] = list(self.review_label.keys())
        d['labels_predicted'] = list(self.review_label.values())
        index_label_df = pd.DataFrame(d)
        # task: join the new df with the test_data by "index"
        # drop the indices from the entire dataset (including train, test)
        test_data = self.test.reset_index(drop=True) 
        # create a new test_index column for joining index_label_df by "index"
        test_data = test_data.reset_index()
        self.joined_test_data = pd.merge(test_data, index_label_df, how = 'left', left_on = 'index', right_on = 'index')
        # output to csv file
        self.joined_test_data.to_csv('../res/labels_predicted_lg.csv', index=False)
        return self.joined_test_data

    def get_review_with_multiple_labels(self, threshold = 2):
        '''set threshold for getting review indices that have assigned labels more than this threshold '''
        indices_with_multiple_labels = [idx for idx in self.review_label.keys() if len(self.review_label[idx]) > threshold]
        print('indices with multiple labels:', indices_with_multiple_labels)
        return indices_with_multiple_labels

    def get_review_data_with_multiple_labels(self,indices_with_multiple_labels):
        ''' get subset dataframe by indices with multiple labels'''
        data_multiple_labels = self.joined_test_data.loc[indices_with_multiple_labels]
        return data_multiple_labels


    





