# Multi-class classification of user reviews

This project aims to build an automatic classifier that can classify user reviews accurately and efficiently into 10 different subclasses defined, including 预订, 出发, 设计, 性能, 到达, 行程, 机上, 计划, 中转, 售后. With such classifier, the company will be able to make improvements and provide better user experience in terms of every aspects of flight experience.

## Codes Explantion
### Python files
* modeling.py:
    - basic modeling code setup, functions include: get_precision, plot_confusion_matrix, get_clf_result, grid_search
    - used with basic_pipeline_setup_v5.ipynb
            
* prepare_data.py:
    - load data; and split data into train, test set
    - preprocess data: countvectorizer() for one-hot encoding and tf-idf transformer() to preprocess data
    - get precision score by comparing y_pred and y_test
    - used with basic_pipeline_setup_v5.ipynb
        
* tokenization.py:
    - used to tokenize Chinese text
    - including remove non-Chinese characters, stopwords, digits, and punctuations
                
* semi-supervise.py:
    - functions include: load_labeled_data(),load_unlabeled_data(),concat_data(),get_X(), label_propagation()
    - mainly referenced from sklearn semi-supervised package

* unlabeled_preprocess.py:
    - preprocess unlabeled data
    - sample unlabeled data

* word2vec.py:
    - use pretrained word vectors to train labeled data

* modeling_main.py:
    - combining modeling.py and prepare_data.py
    - main code blocks for basic_pipeline_setup_v6.ipynb:
    
    
### ipynb files
    
* basic_pipeline_setup_v4.ipynb:
    - update: split tokenization part into a separate python file 'tokenization.py', which can be imported directly
    - update: split data preprocessing part into another separate ipynb file 'data_preprocessing.ipynb'

* data_preprocessing.ipynb:
    - handles tokenization, merge data under different categories into one single file
    - tokenized result stored as'/res/all_labeled_data_v3.csv'
        
* basic_pipeline_setup_v5.ipynb: (MOST RECENT)
    - updated: plot the probability distribution for each label
    - updated: set probability threshold to be median probability of each class
    - TODO: after getting the valid labels (meeting the prob. threshold for each class), how to handle one user review with multiple valid labels --> one thoughts: pick the label with the highest probability among all
    - main codes are based on modeling.py
    - more reference: Choosing Logisitic Regression’s Cutoff Value for Unbalanced Dataset(http://ethen8181.github.io/machine-learning/unbalanced/unbalanced.html)
            
* unlabeled_preprocess.ipynb: 
    - sample unlabeled data (5000) and tokenized text
    - output tokenized unlabeled data is stored under res/tokenized_sampled_unlabeled_reviews
        
* semi_supervised.ipynb: 
    - main codes to implement semi-supervised learning using sklearn packages
    - based on knn algorithm
    - return a dataframe of unlabeled data with review tokens and predicted labels
    - TODO: 
    - idea: graph-based --> compute cosine similarity for each node (each node represent a user review)
    - ideally would be better if can get original untokenized reviews and format into the output dataframe --> for CSAir to review manually

* set_manual_threshold_v2.ipynb: 
    - based on basic_pipeline_setup_v5.ipynb, organize codes: that allows you to pass on a modeling method, e.g., logisticRegression(), and get the classification results after manually setting predicted probability threshold
    - combined in usage with modeling_main.py
         
_(updated: 04/09/2019)_
* set_manual_threshold_v3.ipynb: (MOST RECENT)
    - updated: rewrite functions in set_manual_threshold_v2 into an entire function
    - can pass on: Logistic regression, Naive Bayes 
        
_(updated: 04/08/2019)_
* process_raw_labeled_data.ipynb:
    - based on process_labeled_data.py 
    - used for process raw labeled data from beginning (10 class csv files) to end
    - output data: all_raw_labeled_data.csv (not seperate & tokenize review)
        

### word2vec files
        
word2vec_pretrained folder contains codes focusing on using pretrained word2vec to build model
* pretrained_word2vec.ipynb
    - using '/Source_Data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5' pretrained Chinese word vectors
    - downloaded from: https://github.com/Embedding/Chinese-Word-Vectors (Baidu Encyclopedia 百度百科 300d word)
    - tutorial & code reference: 
    - https://keras-cn-docs.readthedocs.io/zh_CN/latest/blog/word_embedding/ 
    - https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py

* word2vec_v2.ipynb:
    - TODO: try other methods
                    


## Other Codes

* bag_of_words+NN.ipynb:
    - using one-hot encoding and build basic neural network to train 
    - TODO: one-hot encoding process has updated, and so does data preprocessing stage; may need to modify accordingly

* check duplication.ipynb:
    - check if duplication exists in multiple classes; if true, then may consider soft classification 

* process_wiki.py:
    - process wiki Chinese text recourses, and get pre-trained txt file as "Source_Data/wiki.zh.txt"

* unlabeled_preprocess.ipynb:
    - rewrite tokenization preprocessing into class
    - sampled 5000 unlabeled examples from second batch data
    - TODO: combine labeled data and unlabeled data to build corpus for countervectorizer() and tf-idf transformer()

 * basic_pipeline_setup_v3.ipynb:
    - update version of code pipeline 
    - adding more NLP for data preprocessing, e.g. remove stopwords, non-Chinese characters, etc.
    - using countvectorizer() and tf-idf transformer() to construct word vectors
        

## Data

* all_labeled_data(formatted).csv: 
    - reformat original output data to "class","reviews" format

* ../res/labels_predicted_lg.csv:
    - logistic regression after manually setting proba threshold 

* ../res/labeled_data_with_without_tk.csv
    - include review, review_tokenized, label, label_encoded 


### codes can be deprecated or moved to archive: 
* basic_pipeline_setup_v2.ipynb
* basic_pipeline_setup.ipynb
* text_processing.ipynb
* split_data.ipynb
* get_output_data.ipynb - originally intended to separate data preprocessing part from code pipeline setup; can work on that later


### references from online resources:
* 搜狐新闻文本分类：[机器学习大乱斗](https://www.jianshu.com/p/e21b570a6b8a)
* [用户行为分析:用wiki百科中文语料训练word2vec模型](https://blog.csdn.net/hereiskxm/article/details/49664845)

      
   
  
  


