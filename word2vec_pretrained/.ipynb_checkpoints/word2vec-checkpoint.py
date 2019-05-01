import os
import numpy as np
import pandas as pd

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
from keras.layers import Flatten
from keras.layers import Embedding
import tensorflow as tf


class word2vec():
    def __init__(self):
        self.embeddings_index = {}
        self.MAX_SEQUENCE_LENGTH = 1000
        self.MAX_NUM_WORDS = 20000
        self.EMBEDDING_DIM = 100
        self.VALIDATION_SPLIT = 0.2
        self.all_labeled_data = pd.DataFrame()
        self.labels_index = {}
        self.word_index  = {}
        self.texts = np.array([])
        self.labels = np.array([])
        self.data = np.array([])
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.X_val = np.array([])
        self.y_val = np.array([])
        self.embedding_matrix = np.array([])

    def load_pretrained_vectors(self, file_path):
        f = open(file_path)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(self.embeddings_index))
        return self.embeddings_index 

    def prepare_data(self,data_file_path):
        self.all_labeled_data = pd.read_csv(data_file_path)
        self.texts = self.all_labeled_data.review_tokens.values
        self.labels = self.all_labeled_data.label_encoded.values
        
        # get a dictionary that map each original label to its encoded label, e.g., {'中转': 0,...}
        for label in self.all_labeled_data.label.unique().tolist():
            self.labels_index[label] = self.all_labeled_data[self.all_labeled_data['label'] == label]['label_encoded'].unique()[0]

        tokenizer = Tokenizer(nb_words=self.MAX_NUM_WORDS)
        tokenizer.fit_on_texts(self.texts)
        sequences = tokenizer.texts_to_sequences(self.texts)

        self.word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(self.word_index))

        self.data = pad_sequences(sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        print('Shape of data tensor:', self.data.shape)
        print('Shape of label tensor:', self.labels.shape)
        
        # Converts a class vector (integers) to binary class matrix
        self.labels = to_categorical(np.asarray(self.labels))
        print('Shape of data tensor:', self.data.shape)
        print('Shape of label tensor:', self.labels.shape)

        # split the data into a training set and a validation set
        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)
        self.data = self.data[indices]
        self.labels = self.labels[indices]
        nb_validation_samples = int(self.VALIDATION_SPLIT * self.data.shape[0])

        self.X_train = self.data[:-nb_validation_samples]
        self.y_train = self.labels[:-nb_validation_samples]
        self.X_val = self.data[-nb_validation_samples:]
        self.y_val = self.labels[-nb_validation_samples:]
        return  self.X_train, self.y_train, self.X_val, self.y_val
    
    def train_data(self):
        # 据得到的字典生成上文所定义的词向量矩阵
        self.embedding_matrix = np.zeros((len(self.word_index) + 1, self.EMBEDDING_DIM))
        for word, i in self.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            
        # 将这个词向量矩阵加载到Embedding层
        embedding_layer = Embedding(len(self.word_index) + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.MAX_SEQUENCE_LENGTH,
                                    trainable=False)
        
        # 使用一个小型的1D卷积解决分类问题
        sequence_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(35)(x)  # global max pooling
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(len(self.labels_index), activation='softmax')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
        model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val),
                  nb_epoch=2, batch_size=128)
        return model