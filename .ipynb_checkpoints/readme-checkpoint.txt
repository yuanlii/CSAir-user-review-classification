readme: 

# working progress mainly includes data preprocessing and basic code pipeline set up. Code files include:
    - unlabeled_preprocess.ipynb:
         - rewrite tokenization preprocessing into class
         - sampled 5000 unlabeled examples from second batch data
         - TODO: combine labeled data and unlabeled data to build corpus for countervectorizer() and tf-idf transformer()

     - process_wiki.py:
         - process wiki Chinese text recourses, and get pre-trained txt file as "Source_Data/wiki.zh.txt"

    - basic_pipeline_setup_v3.ipynb:
          - update version of code pipeline 
          - adding more NLP for data preprocessing, e.g. remove stopwords, non-Chinese characters, etc.
          - using countvectorizer() and tf-idf transformer() to construct word vectors
    
    - check duplication.ipynb:
          - check if duplication exists in multiple classes; if true, then may consider soft classification 


# also work on word2vec using pretrained model and Neural Network. Codes stored in ** word2vec_pretrained ** include:

    - pipeline_setup_word2vec_v2.ipynb
        - TODO: using cnews data (see **Source_Data/语料集**) to train word2vec model; using pretrained model to do word2vec embedding
        
    - pipelie_setup_word2vec.ipynb
        - TODO: basically same as v2 file; had some extra reference codes
        
    - pretrained_word2vec.ipynb: 
        - TODO: embedding wiki pretrained models into word2vec model
    
    - word2vec model training output:(using cnews data)
        - word_vector_cnn.h5
        - word2vec.model: model can be loaded to train word2vec embedding of new documents
        - word2vec.model.trainables.syn1neg.npy: MISC
        - word2vec.model.wv.vectors.npy: MISC
        

# other explanation:
    - bag_of_words+NN.ipynb:
        - using one-hot encoding and build basic neural network to train 
        - TODO: one-hot encoding process has updated, and so does data preprocessing stage; may need to modify accordingly

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
          
      
    
  
  


