from mlxtend.classifier import StackingCVClassifier
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold
from sklearn.metrics import classification_report



# data

data = pd.read_csv(r"../data/processed/clean_train_data.csv",
        usecols = ['comment','label'], encoding="utf-8")
data = data.dropna()


## TF-IDF + SVD
clf = Pipeline([
                ('tfidf', TfidfVectorizer(max_df=0.3, min_df=10, ngram_range=(1, 2), norm="l2")),
                ('svd', TruncatedSVD(n_components = 500, random_state=42)),
                ])
# Sample data - 25% of data to test set
train, test = train_test_split(data, random_state=1, test_size=0.25, shuffle=True)

X_train = train["comment"]
Y_train = train["label"]
X_test = test["comment"]
Y_test = test["label"]


# transform each sentence to numeric vector with tf-idf value as elements
X_train_vec = clf.fit_transform(X_train)
X_test_vec = clf.transform(X_test)
Y_train = Y_train.values
Y_test = Y_test.values


## dump clf
# print("dump pipeline")
# pickle.dump(clf, open(r"models/tfidf.pkl", "wb"))
# Initializing Classifiers
SEED = 42
clf1 = LogisticRegression(random_state= SEED)
clf2 = DecisionTreeClassifier(random_state= SEED)
clf3= RandomForestClassifier(random_state= SEED)
clf4 = GaussianNB()
clf5 = BernoulliNB()

lr = LogisticRegression(random_state= 100)

sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3, clf4, clf5],
                            use_probas=True,
                            meta_classifier=lr,
                            random_state=42)

# print('5-fold cross validation:\n')
# kf = StratifiedKFold( n_splits= 5, random_state=SEED, shuffle=True)

# for clf, label in zip([clf1, clf2, clf3,clf4, clf5, sclf], 
#                       [' LogisticRegression', 
#                        'DecisionTreeClassifier', 
#                        'RandomForestClassifier',
#                        'GaussianNB',
#                        'BernoulliNB',
#                        'StackingClassifier']):

#     scores = cross_val_score(clf, X_train_vec, Y_train, cv=kf, scoring='f1')
#     print("F1 score: %0.4f (+/- %0.4f) [%s]" 
#           % (scores.mean(), scores.std(), label))

sclf.fit(X_train_vec, Y_train)
# pred =  sclf.predict(X_test_vec)
print("start dump")
# pickle.dump(sclf, open(r'models/stacking_mlxtend.sav', 'wb'))
print("done ")
# print("report", classification_report(Y_test, pred))