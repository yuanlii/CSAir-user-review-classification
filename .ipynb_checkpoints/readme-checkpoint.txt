readme: 
    
    This project is working to classify user reviews into 10 different categories.

    Codes Explantion
    -----------------
    
        - basic_pipeline_setup_v4.ipynb:
            - update: split tokenization part into a separate python file 'tokenization.py', which can be imported directly
            - update: split data preprocessing part into another separate ipynb file 'data_preprocessing.ipynb'

        - data_preprocessing.ipynb:
            - handles tokenization, merge data under different categories into one single file
            - tokenized result stored as'/res/all_labeled_data_v3.csv'

        - tokenization.py:
            - used to tokenize Chinese text, including:
                - including remove non-Chinese characters, stopwords, digits, and punctuations
        
        - word2vec_pretrained folder contains codes focusing on using pretrained word2vec to build model
            - pretrained_word2vec.ipynb
                - using '/Source_Data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5' pretrained Chinese word vectors
                    - downloaded from: https://github.com/Embedding/Chinese-Word-Vectors (Baidu Encyclopedia 百度百科 300d word)
                - tutorial & code reference: 
                    - https://keras-cn-docs.readthedocs.io/zh_CN/latest/blog/word_embedding/ 
                    - https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
                    
       

    Other Codes
    -----------
        - bag_of_words+NN.ipynb:
            - using one-hot encoding and build basic neural network to train 
            - TODO: one-hot encoding process has updated, and so does data preprocessing stage; may need to modify accordingly

        - check duplication.ipynb:
            - check if duplication exists in multiple classes; if true, then may consider soft classification 
            
        - process_wiki.py:
            - process wiki Chinese text recourses, and get pre-trained txt file as "Source_Data/wiki.zh.txt"

        - unlabeled_preprocess.ipynb:
            - rewrite tokenization preprocessing into class
            - sampled 5000 unlabeled examples from second batch data
            - TODO: combine labeled data and unlabeled data to build corpus for countervectorizer() and tf-idf transformer()

         - basic_pipeline_setup_v3.ipynb:
            - update version of code pipeline 
            - adding more NLP for data preprocessing, e.g. remove stopwords, non-Chinese characters, etc.
            - using countvectorizer() and tf-idf transformer() to construct word vectors
        


# codes can be deprecated or moved to archive: 
      - basic_pipeline_setup_v2.ipynb
      - basic_pipeline_setup.ipynb
      - text_processing.ipynb
      - split_data.ipynb
      - get_output_data.ipynb: 
          - originally intended to separate data preprocessing part from code pipeline setup; can work on that later
          

# references from online resources:
    - 搜狐新闻文本分类：机器学习大乱斗(https://www.jianshu.com/p/e21b570a6b8a)
    - 【用户行为分析】 用wiki百科中文语料训练word2vec模型(https://blog.csdn.net/hereiskxm/article/details/49664845)
          
      
   
  
  


