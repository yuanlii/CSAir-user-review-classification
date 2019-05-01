import pandas as pd
import numpy as np
import random
import os
os.chdir('/Users/liyuan/desktop/CSAir/codes')
import fastText 

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

from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from pandas_ml import ConfusionMatrix

from semi_supervise import Semi_Supervise

# ---- prepare data from user reviews to train FastText model ----
# load data
ss = Semi_Supervise()
labeled_data = ss.load_labeled_data('../res/labeled_data_with_without_tk.csv')
# load unlabeled data
# unlabeled_data = ss.load_unlabeled_data_csv('../res/unlabeled_review_5000.csv')

# load more unlabeled data ..
unlabeled_data = ss.load_unlabeled_data_csv('../res/unlabeled_review_反馈数据.csv')
# concatenate labeled and unlabeled data
data_concat = ss.concat_data()
data_concat.head()

# output data to txt file => data would be feed into fasttext
train_data = data_concat.review_tokens
train_data.to_csv('../res/sampled_data_fasttext.txt', index = False)

# Finished: locally train fasttext using labeled + unlabeled data (5000)
# use command below to locally train skip-gram fasttext model: 
# ./fasttext skipgram -input ../fasttext_train_data/sampled_data_fasttext.txt -output ../fasttext_train_data/model

# train cbow version:
# ./fasttext cbow -input ../fasttext_train_data/sampled_data_fasttext.txt -output ../fasttext_train_data/model_cbow


class fasttext():
    def __init__(self):
        self.embeddings_index = {}
        self.MAX_SEQUENCE_LENGTH = 1000
        self.MAX_NUM_WORDS = 20000
        self.EMBEDDING_DIM = 300
        self.VALIDATION_SPLIT = 0.2
        self.labels_index = {}
        self.word_index  = {}
    
    def load_pretrained_model(self, model_path):
        model_pretrained = fastText.load_model(model_path) 
        return model_pretrained
    
    def prepare_data(self,data_file_path):
        self.all_labeled_data = pd.read_csv(data_file_path)
        self.texts = self.all_labeled_data.review_tokens.astype('str').values
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
        self.indices = np.arange(self.data.shape[0])
        np.random.shuffle(self.indices)
        self.data = self.data[self.indices]
        self.labels = self.labels[self.indices]
        nb_validation_samples = int(self.VALIDATION_SPLIT * self.data.shape[0])

        self.X_train = self.data[:-nb_validation_samples]
        self.y_train = self.labels[:-nb_validation_samples]
        self.X_val = self.data[-nb_validation_samples:]
        self.y_val = self.labels[-nb_validation_samples:]
        return  self.X_train, self.y_train, self.X_val, self.y_val
    
    def get_embedding_matrix(self):
        # 据得到的字典生成上文所定义的词向量矩阵
        embedding_matrix = np.zeros((len(self.word_index) + 1, self.EMBEDDING_DIM))
        for word, i in self.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            # updated:
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
    
    def setup_neural_net(self):
        # get word embedding matrix
        self.embedding_matrix = self.get_embedding_matrix()

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
        return sequence_input,preds
    
    
    def train_data(self,X_train,y_train,X_val,y_val):
        sequence_input,preds = self.setup_neural_net()
        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
        # can change the number of epoch accordingly
        # 7 generates the best performance
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  nb_epoch=7, batch_size=128)  
        
        # evaluate model using model.evaluate()
        scores = model.evaluate(X_val, y_val, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        
        # get predicted class label
        self.output = model.predict(X_val)
        predicted_label_list = self.get_pred_label(self.output)
        return predicted_label_list
    
    
    def get_pred_label(self,output):
        '''get predicted class label based on prediction output'''
        predicted_label_list = []
        for i in range(len(output)):
            predicted_label = output[i].argmax(axis=-1)
            predicted_label_list.append(predicted_label)        
        return predicted_label_list
    
    
    def incorporate_pred_label(self):
        '''return prediction results back to df'''
        # indices is a numpy array, need to convert to a list of indices before feed into df to get sub df 
        # recreate df based on the shuffled indices
        indices = self.indices
        all_labeled_data = self.all_labeled_data.iloc[list(self.indices)]
        nb_validation_samples = int(self.VALIDATION_SPLIT * self.data.shape[0])
        print(nb_validation_samples)
        # need to get the indices of the validation data
        train_val_bound = self.data.shape[0] - nb_validation_samples
        # get validation dataset
        val_df = all_labeled_data[train_val_bound:]
        return val_df

    def map_label(self,df,predicted_label_list):
        '''map predicted labels to original class'''
        # print(predicted_label_list[:10])
        label_dct = self.labels_index
        df['pred_label_encodes'] = predicted_label_list
        # get reversed labels_index dictionary
        reversed_label_dct = {}
        for i in range(len(label_dct)):
            reversed_label_dct[list(label_dct.values())[i]] = list(label_dct.keys())[i]

        # map predicted labels
        pred_label = [reversed_label_dct.get(label) for label in predicted_label_list]
        df['pred_label'] = pred_label
        return df
    
    
    def evaluate_performance(self,val_df):
        # evaluate performance
        y_val_true = val_df.label.values
        y_val_pred = val_df.pred_label.values
        self.get_confusion_matrix(y_val_true,y_val_pred) 
        
    
    def get_confusion_matrix(self,y_test,y_pred):
        '''get tp,tn,fp,fn for each class'''
        cm = ConfusionMatrix(y_test, y_pred)
        cm.print_stats()
        
        
    def over_sampling(self):
        '''modeling after over sampling'''
        smote = SMOTE('minority')
        X_train_sm, y_train_sm = smote.fit_sample(self.X_train,self.y_train)
        print(X_train_sm.shape, y_train_sm.shape)
        
        # fit model based on new data set
        predicted_label_list = self.train_data(X_train_sm,y_train_sm,X_val,y_val)
        return predicted_label_list
    

    ft = fasttext()
    # model = ft.load_pretrained_model('fasttext_train_data/model.bin')
    # model = ft.load_pretrained_model('fasttext_train_data/model_cbow.bin')

    # load pretrained from official site ..
    model = ft.load_pretrained_model('../Source_Data/cc.zh.300.bin')


    # get more model methods
    print('dimention of word vector:',model.get_dimension())
    # load word vector
    model.get_word_vector('航班').astype('float32')


    X_train, y_train, X_val, y_val = ft.prepare_data('../res/labeled_data_with_without_tk.csv')

    predicted_label_list = ft.train_data(X_train,y_train,X_val,y_val)

    # check label index
    print('label encoding dictionary:', ft.labels_index)

    val_df = ft.incorporate_pred_label()
    val_df = ft.map_label(val_df,predicted_label_list)
    val_df.head()

    # evaluate performance
    ft.evaluate_performance(val_df)