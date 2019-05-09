# CSAir User review classification

## Project Goal

User experience has been a cruicial part in contributing to a company's success. Partnered with China Southern Airline, we aim to build a auto-classifier that can successfully classify user reviews into 10 differernt labels, which are associated with different functional teams of the company. User reviews for this project are collected across a wide range of digital platforms, including iOS app store, Google Play App, Huawei App store as well as user reviews collected directly from its own released app. Classifiers built for this project will be integrated into the company's operation, in order to gain better understanding of customer needs and provide insights for the company's next step improvement.

## Data Preprocessing
* loaded user review data from separate txt files, and concatenated them into one single data file
* data cleaning: remove non-Chinese characters, stopwords, digits, and punctuations
* Chinese word segmentation: used [Jieba 结巴中文分词](https://github.com/fxsjy/jieba) for segmentation of Chinese words, e.g.,
  我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学
* exploratory data analysis: distribution among the 10 labels, we can tell certain unbalance exists

![labeled_data_dist](https://github.com/yuanlii/CSAir_user_review_classification/blob/master/images/labeled_data_dist.png)



## Bag of words + TF-IDF transformer
* Implemented Bag-of-Words: convert user reviews into one-hot encoding using countvectorizer() and TF-IDF transformation
* Split user review data into training and test set


## Build Classifier using supervised method

Overall, for multiclass classification problem in general, suppose we have N classes, for each of the N classes, we would want to predict using 

![multiclass classification formula](https://github.com/yuanlii/CSAir_user_review_classification/blob/master/images/multiclass_classification_formula.png)

where pi(x) is the density for each of the N classes.

### 1. vectorized method
* use scikit learn multiclass classifier to directly predict for 10 class labels
* TODO: more explanation

### 2. One-vs-All Classification
* Build N (in our case is 10) different binary classifiers. For the ith classifier, let the positive examples be all the points in class i, and let the negative examples be all the points not in class i. 

![Andrew Ng Machine Learning course screenshot](https://github.com/yuanlii/CSAir_user_review_classification/blob/master/images/multiclass_screenshot.png)

* Building binary classifier for each label class
  * use Grid Search and Cross Validation for model selection
   * single modeling method + multiple hyperparameter tuning (with demonstrative best performance for each label class); an example is shown below, in which I chose SVC for modeling method.
   ![CV best params for SVC](https://github.com/yuanlii/CSAir_user_review_classification/blob/master/images/multiclass_crossvalidation_best_params.png)

After building binary classifier for each class, I stacked the predicted probability horizontally (column-wise). P(X,Y) is the predicted probability of review X being classified in class Y, e.g, P(0,0) is the predicted probability of review 0 being classified in class 0 (in this case, "计划“）

![proba_matrix_sketch](https://github.com/yuanlii/CSAir_user_review_classification/blob/master/images/proba_matrix_sketch.jpg)

The actual probability matrix that I managed to generate looks like below: 
![proba_matrix](https://github.com/yuanlii/CSAir_user_review_classification/blob/master/images/proba_matrix.jpg)

* distribution of predicted probability in each label class
  * with the chosen modeling method, e.g., Logistic Regression, SVC, Multinomial NB, etc., we can get the predictive probability distribution of each label class. An example is given below:

![predictive proba distribution of each label class](https://github.com/yuanlii/CSAir_user_review_classification/blob/master/images/proba_distribution_by_class_lg.png)
  

* Manually set threshold for predicted probability

Since probability distribution among classes varies quite a lot, it would not make sense to use the default 0.5 level to classify user review. Below is an example of the distribution of the predicted probability of label class 1 by using Logistic Regression, the majority probability lies on the left of 0.5 level. In such case, our system can hardly picked any label 1 review. 

![proba dist for class1](https://github.com/yuanlii/CSAir_user_review_classification/blob/master/images/proba_dist_class1.jpg)

In order to avoid such cases, we decided to manually set the probability threshold in terms of deciding whether a piece of user review to be classified as positive in certain label class or not. The tricky part is, how should we set the probability threshold for label classes differently?

First we would compute and get "class priors" - ratios of size of each label class divided by total number of user reviews. After getting the predicted probability of each review in each class, we would use the upper-bound percentile as the threshold in deciding whether the review is classified as positive or not. Let's take an example. If there are 1700 user reviews in total and 149 reviews labeled as class 0, the class prior of such class (class 0) should be 149/561. Ideally we would want to pick reviews and label them to maintain the balance of different data size of each label. If there are 561 reviews in test data, the proportional labeled review for us to pick in test data should be approximately 561 * (149/17000).

![manual_threshold](https://github.com/yuanlii/CSAir_user_review_classification/blob/master/images/prob_dict_class1_manual.jpg)

code extractions for manually setting threshold: (more please see prepare_data.py)

``` python
percentile = (1 - self.class_priors[i])*100
threshold = np.percentile(class_prob, percentile) 
```


## Word2vec 
* used Baidu Baike (百度百科）word2vec pre-trained model
  * size: ~4.6GB
  * [download link](https://github.com/Embedding/Chinese-Word-Vectors)

* built word embedding matrix leveraging Baidu Baike pretrained word vectors
* incorporated inforation from embedding mastrix and added embedding layer in CNN model 
  * used Relu activation for multiple layers in CNN, and softmax activation for the last layer since we have more than 2 classes to classify
  * used categorical_crossentropy as loss function

code extractions for manually setting probability threshold: (more to see in prepare_data.py)

```python
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
```

## More Exploring
* BERT:
  * Learn [BERT](https://github.com/google-research/bert) for word embedding and train data
* semi-supervised learning 
  * using sklearn label propagation, based on all labeled data and 5000 unlabeled data
   * applied kernel = knn and kernel = rbf (with prediction result output as csv file in res folder)
   * files:
    * semi_supervised.ipynb => kernel = 'knn'
    * semi_supervised.ipynb => kernel = 'rbf'

* Unspervised learning (clustering results)

Plot the kmeans clustering results(by initializing 10 groups), after applied PCA and get two principal components. The diagram shows that kmeans clustering cannot very well capture the boundary of data.

![K-means clustering](https://github.com/yuanlii/CSAir_user_review_classification/blob/master/images/clustering_kmeans.png)


## Acknowledgments
* [Machine Learning — Multiclass Classification with Imbalanced Dataset](https://towardsdatascience.com/machine-learning-multiclass-classification-with-imbalanced-data-set-29f6a177c1a)
* [reference: Multiclass Classification - MIT](www.mit.edu/~9.520/spring09/Classes/multiclass.pdf)
* 搜狐新闻文本分类：[机器学习大乱斗](https://www.jianshu.com/p/e21b570a6b8a)
* [用户行为分析 用wiki百科中文语料训练word2vec模型](https://blog.csdn.net/hereiskxm/article/details/49664845)


## Resources backup
### 中文语料库
* [(译)吊炸天的中文自然语言处理工具和语料库介绍](https://mlln.cn/2018/06/02/[%E8%BD%AC]%E5%90%8A%E7%82%B8%E5%A4%A9%E7%9A%84%E4%B8%AD%E6%96%87%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E5%B7%A5%E5%85%B7%E5%92%8C%E8%AF%AD%E6%96%99%E5%BA%93%E4%BB%8B%E7%BB%8D/)
  * 介绍各种中文语料库
* [SentiBridge](https://github.com/rainarch/SentiBridge)
  * 关于点评资料

          
