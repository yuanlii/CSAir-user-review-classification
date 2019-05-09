# CSAir User review classification

## Project Goal

User experience has been a cruicial part in contributing to a company's success. Partnered with China Southern Airline, we aim to build a auto-classifier that can successfully classify user reviews into 10 differernt labels, which are associated with different functional teams of the company. User reviews for this project are collected across a wide range of digital platforms, including iOS app store, Google Play App, Huawei App store as well as user reviews collected directly from its own released app. Classifiers built for this project will be integrated into the company's operation, in order to gain better understanding of customer needs and provide insights for the company's next step improvement.

## Supervised Learning

Overall, for multiclass classification problem in general, suppose we have N classes, for each of the N classes, we would want to predict using 

![multiclass classification formula](https://github.com/yuanlii/CSAir_user_review_classification/blob/master/images/multiclass_classification_formula.png)


1. vectorized method
    * use scikit learn multiclass classifier to directly predict for 10 class labels

2. One-vs-All Classification
    * Build N (in our case is 10) different binary classifiers. For the ith classifier, let the positive examples be all the points in class i, and let the negative examples be all the points not in class i. 

![Andrew Ng Machine Learning course screenshot](https://github.com/yuanlii/CSAir_user_review_classification/blob/master/images/multiclass_classification.png)



## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
* [reference: Multiclass Classification - MIT](www.mit.edu/~9.520/spring09/Classes/multiclass.pdf)
