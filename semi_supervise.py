import os
os.chdir('/Users/liyuan/desktop/CSAir/codes')
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.semi_supervised import label_propagation

from tokenization import Tokenization
from prepare_data import PrepareData

class Semi_Supervise():
    def __init__(self):
        self.labeled_data = pd.Series()
        self.unlabeled_data = pd.Series()
        self.data_concat = pd.Series()
        # initialize matrix, need to parse into a empty list inside np.array()
        self.X = np.array([])
    
    def load_labeled_data(self, labeled_data_path):
        data_p = PrepareData()
        self.labeled_data = data_p.load_data(labeled_data_path)
        self.labeled_data = self.labeled_data[['review_tokens','label_encoded']]
        print('there are %d examples in labeled dataset'%len(self.labeled_data))
        return self.labeled_data

    def load_unlabeled_data(self, unlabeled_data_path):
        # load unlabeled data
        unlabeled_lst = []
        with open(unlabeled_data_path, 'r', encoding='utf-8') as input_file:
            unlabeled_lst +=  input_file.readlines()
        # reformat unlabeled data into dataframe
        d = {}
        d['review_tokens'] = unlabeled_lst
        d['label_encoded'] = np.nan
        self.unlabeled_data = pd.DataFrame(d)
        return self.unlabeled_data

    def concat_data(self):
        '''concatenate labeled and unlabeled data'''
        self.data_concat = pd.concat([self.labeled_data, self.unlabeled_data],ignore_index=True)
        return self.data_concat

    def get_X(self):
        '''convert text data to digits'''
        count_v0= CountVectorizer()
        counts_all = count_v0.fit_transform(self.data_concat['review_tokens'])
        count_v1= CountVectorizer(vocabulary=count_v0.vocabulary_)  
        # implement tf-idf
        tfidftransformer = TfidfTransformer()
        self.X = tfidftransformer.fit(counts_all).transform(counts_all)
        return self.X
    
    def label_propagation(self):
        rng = np.random.RandomState(0)
        indices = np.arange(len(self.data_concat))
        rng.shuffle(indices)

        # there are 1700 labeled training examples
        X = self.X.toarray() # convert sparse to dense
        y = self.data_concat.label_encoded.values

        n_total_samples = len(y)
        n_labeled_points = 1700

        indices = np.arange(n_total_samples)
        unlabeled_set = indices[n_labeled_points:]
        print(unlabeled_set)

        # Shuffle everything around
        y_train = np.copy(y)
        # assign '-1' as label_encoded to unlabeled data
        y_train[unlabeled_set] = -1
        print('y_train:', y_train)

        # TODO: need to parse into vectorized data
        # Learn with LabelSpreading
        # lp_model = label_propagation.LabelSpreading(kernel = 'knn', gamma=0.25, max_iter=5)
        lp_model = label_propagation.LabelSpreading(kernel = 'knn')
        lp_model.fit(X, y_train)
        predicted_labels = lp_model.transduction_[unlabeled_set]
        true_labels = y[unlabeled_set]
        print('true labels: ', true_labels)
        print('predicted labels:', predicted_labels)
        return predicted_labels

    
    def match_reviews_to_pred_labels(self, encode_dict_path):
        # match predicted labels to reviews
        self.unlabeled_data['predicted_labels'] = predicted_labels
        with open(encode_dict_path,'r') as f:
            labels_encoded_dict = json.load(f)
        def get_class(row):
            for k in list(labels_encoded_dict.keys()):
                if labels_encoded_dict[k] == row:
                    return k  
        self.unlabeled_data['predicted_class'] = unlabeled_data['predicted_labels'].apply(get_class)
        return self.unlabeled_data

    
