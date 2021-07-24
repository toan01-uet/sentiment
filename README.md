# Sentiment Analysis
## Demo
![demo](https://media.giphy.com/media/2U0On60uHzaoprTaVK/giphy.gif)
## Description
 
In this project, we aim to develop a system to automatically classifier incoming message, comment,.. and tells whether the underlying sentiment is positive or negative.

We used stacking ensemble (navie bayes, logistic regression, decision tree, randomn forest) for sentiment classification using AIViVN's comments dataset.

The model scored 0.87 (f1 score) in test set.

## Model
We used [mlxtend](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/#example-2-using-probabilities-as-meta-features) to stacking models. 

The first-level classifiers are: 

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. GaussianNB 
5. BernoulliNB

The meta classifier is Logistic Regression.  The class-probabilities of the first-level classifiers are used to train the meta-classifier.

Here’s what an ensemble stacking model does:
![sys]()
<!-- ## Data
The dataset contains 16087 comments:

- 9253 positive comments (label 0) 
- 6796 negative commnents (label 1) -->




