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

class Modeling():
    def __init__(self,X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
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
    

    