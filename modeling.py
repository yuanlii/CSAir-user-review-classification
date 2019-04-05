from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB  
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
import keras
from keras.utils import to_categorical

import pandas as pd
import os
import re
import glob
import numpy as np
import time
import jieba
import jieba.posseg as pseg
import jieba.analyse
import matplotlib.pyplot as plt
import seaborn as sns

class Modeling():
    def __init__(self,X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        # define a dictionary, key --> class, values --> class labels that meet the prob. threshold
        self.label_lst = {}
        self.review_labels = {}
        
        
    def get_precision(self,y_pred):
        '''this function returns a precision score for the model'''
        num = 0
        y_pred = y_pred.tolist()
        for i,pred in enumerate(y_pred):
            if int(pred) == int(self.y_test.values[i]):
                num += 1
        precision = float(num) / len(y_pred)
        #print('precision: '+'{:.2f}'.format(precision))
        return precision
    
    def plot_confusion_matrix(self,y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        # Only use the labels that appear in the data
        # classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        # print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               #xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax
    
    
    def get_clf_result(self,model):
        clf = model   
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        result = classification_report(self.y_test, y_pred)
        print('performance of classifier:')
        print(result)
        # cm = confusion_matrix(self.y_test, y_pred)
        # print('result of confusion matrix:')
        # print(cm)
        plt.figure()
        self.plot_confusion_matrix(y_pred)
        
        # use precision as evaluation metrics
        precision = self.get_precision(y_pred)
        return precision
    
    def grid_search(self,model, parameters):
        # use "f1_weightes" as evaluation metrics
        clf = GridSearchCV(model, parameters, cv=5, scoring = 'f1_weighted')
        clf.fit(self.X_train, self.y_train)
        print('best parameters of clf are: ')
        return clf.best_params_

    # updated: 03/30/2019
    def get_label_prob(self, model):
        '''plot probability distribution for each label'''
        # y need to be converted to matrix
        y_train_transformed = to_categorical(self.y_train, dtype='float32')
        y_test_transformed = to_categorical(self.y_test, dtype='float32')

        # modeling --> get probability score for each label  
        multi_target_clf = MultiOutputClassifier(model, n_jobs=-1)
        multi_target_clf.fit(self.X_train, y_train_transformed)
        scores = multi_target_clf.predict_proba(self.X_test)
        
        print('there are %d classes'%len(scores))
        # probability of predicting as class i (0~9) (per review)
        plt.figure(figsize = (10,8))
        for i in range(len(scores)):
            plt.subplot(4, 3, i+1)
            plt.subplots_adjust(top = 0.99, bottom=0.1, hspace=0.5, wspace=0.3)
            # 0 means not predicted as this label; and 1 means predicted as this label, which is what we care about
            prob_class = scores[i][:,1]
            sns.distplot(prob_class)
            plt.title('prob. distribution of class %d'%i)
        return scores

    # each label can be treated as a binary prediction problem: each can compute precision, recall, f1-score accordingly
        # can be referenced: http://ethen8181.github.io/machine-learning/unbalanced/unbalanced.html
    # TODO: reference from classification report below, add confusion matrix (tp,fp,fn,fp) as a new column

    def gen_label_dct(self,scores,threshold_dct):
        '''generate label for each review based on probability and threshold;
        return a dictionary, key: class, value: list of 0-1 labels; 0 means not labeling, 1 means labeling, e.g., {0:[0,1,1,0,0,1,...], 1:[0,0,1,1,0,...], ...} '''
        # there are 10 classes in total
        class_label_dct = {}
        for i in range(10):
            class_labels = []
            for score in scores[i][:,1]:
                if score > threshold_dct[i]:
                    label = 1
                else:
                    label = 0
                class_labels.append(label)
            class_label_dct[i] = class_labels
        return class_label_dct
    
    def map_label_to_review(self, class_label_dct):
        '''get a dictionary listing the class number as keys, and the list of review indices that are classified as high probability '''
        class_reviews_dct = {}
        for i in range(10):
            # extract indices where label = 1
            class_reviews = [ idx for idx in range(len(class_label_dct[i])) if class_label_dct[i][idx] == 1  ]        
            class_reviews_dct[i] = class_reviews
        return class_reviews_dct


    # TODO:
    def get_precision(self):
        pass

