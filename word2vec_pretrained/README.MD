# CSAir User review classification

## Project Goal

User experience has been a cruicial part in contributing to a company's success. Partnered with China Southern Airline, we aim to build a auto-classifier that can successfully classify user reviews into 10 differernt labels, which are associated with different functional teams of the company. User reviews for this project are collected across a wide range of digital platforms, including iOS app store, Google Play App, Huawei App store as well as user reviews collected directly from its own released app. Classifiers built for this project will be integrated into the company's operation, in order to gain better understanding of customer needs and provide insights for the company's next step improvement.

## Supervised Learning
1. vectorized method - use scikit learn multiclass classifier to directly predict for 10 class labels
    * one vs. rest: treating each target label and the rest labels into two groups

2. build 10 classifiers for each class label, using logistic regression and Multinomial Naive Bayes



## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
