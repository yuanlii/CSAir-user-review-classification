# CSAir User review classification

## Project Goal

User experience has been a cruicial part in contributing to a company's success. Partnered with China Southern Airline, we aim to build a auto-classifier that can successfully classify user reviews into 10 differernt labels, which are associated with different functional teams of the company. User reviews for this project are collected across a wide range of digital platforms, including iOS app store, Google Play App, Huawei App store as well as user reviews collected directly from its own released app. Classifiers built for this project will be integrated into the company's operation, in order to gain better understanding of customer needs and provide insights for the company's next step improvement.

### Data Preprocessing
* loaded user review data from separate txt files, and concatenated them into one single data file
* data cleaning: remove non-Chinese characters, stopwords, digits, and punctuations
* Chinese word segmentation: used [jieba](https://github.com/fxsjy/jieba) for segmentation of Chinese words, e.g.,
  我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学
* exploratory data analysis: distribution among the 10 labels, we can tell certain unbalance exists

![labeled_data_dist](https://github.com/yuanlii/CSAir_user_review_classification/blob/master/images/labeled_data_dist.png)



### Bag of words + TF-idf transformer
* Implemented Bag-of-Words: convert user reviews into one-hot encoding using countvectorizer() and TF-IDF transformation
* Split user review data into training and test set


## Supervised Learning

Overall, for multiclass classification problem in general, suppose we have N classes, for each of the N classes, we would want to predict using 

![multiclass classification formula](https://github.com/yuanlii/CSAir_user_review_classification/blob/master/images/multiclass_classification_formula.png)

where pi(x) is the density for each of the N classes.

### 1. vectorized method
* use scikit learn multiclass classifier to directly predict for 10 class labels
* TODO: more explanation

### 2. One-vs-All Classification
* Build N (in our case is 10) different binary classifiers. For the ith classifier, let the positive examples be all the points in class i, and let the negative examples be all the points not in class i. 

![Andrew Ng Machine Learning course screenshot](https://github.com/yuanlii/CSAir_user_review_classification/blob/master/images/multiclass_classification.png)

* Building binary classifier for each label class
  * use Grid Search and Cross Validation for model selection
   * single modeling method + multiple hyperparameter tuning (with demonstrative best performance for each label class); an example is shown below, in which I chose SVC for modeling method.
   ![CV best params for SVC](https://github.com/yuanlii/CSAir_user_review_classification/blob/master/images/multiclass_crossvalidation_best_params.png)

After building binary classifier for each class, I stacked the predicted probability horizontally (column-wise). P(X,Y) is the predicted probability of review X being classified in class Y, e.g, P(0,0) is the predicted probability of review 0 being classified in class 0 (in this case, "计划“）

![proba_matrix_sketch](https://github.com/yuanlii/CSAir_user_review_classification/blob/master/images/proba_matrix_sketch.jpg)

The actual probability matrix that I managed to generate looks like below: 
![proba_matrix](https://github.com/yuanlii/CSAir_user_review_classification/blob/master/images/proba_matrix.jpg)


## word-embedding + word2vec
* TODO



## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
* [reference: Multiclass Classification - MIT](www.mit.edu/~9.520/spring09/Classes/multiclass.pdf)
