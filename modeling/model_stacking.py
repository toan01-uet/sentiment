import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import recall_score, precision_score,f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
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

from sklearn.model_selection import StratifiedKFold
ntrain = X_train_vec.shape[0]
ntest = X_test_vec.shape[0]
SEED = 42 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = StratifiedKFold( n_splits= NFOLDS, random_state=SEED, shuffle=True)


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict_proba(x_te)[:,1]
        oof_test_skf[i, :] = clf.predict_proba(x_test)[:,1]

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

if __name__ == '__main__':
    ## First-Level Models
    lr = LogisticRegression(random_state= SEED)
    dtc = DecisionTreeClassifier(random_state= SEED)
    rfc = RandomForestClassifier(random_state= SEED)
    gau_nb = GaussianNB()
    ber_nb = BernoulliNB()
    # Create our OOF train and test predictions. These base results will be used as new features
    lr_oof_train, lr_oof_test = get_oof(lr, X_train_vec, Y_train, X_test_vec) # LogisticRegression
    dtc_oof_train, dtc_oof_test = get_oof(dtc, X_train_vec, Y_train, X_test_vec) # DecisionTreeClassifier
    rfc_oof_train, rfc_oof_test = get_oof(rfc, X_train_vec, Y_train, X_test_vec) # RandomForestClassifier
    gau_oof_train, gau_oof_test = get_oof(gau_nb, X_train_vec, Y_train, X_test_vec) # GaussianNB
    ber_oof_train, ber_oof_test = get_oof(ber_nb, X_train_vec, Y_train, X_test_vec) # BernoulliNB
    print("Training is complete")

    ## Second-Level Models (Meta models: logisticRe)
   
    x_train = np.concatenate(( lr_oof_train, dtc_oof_train, rfc_oof_train, gau_oof_train, ber_oof_train), axis=1)
    x_test = np.concatenate(( lr_oof_test, dtc_oof_test, rfc_oof_test, gau_oof_test, ber_oof_test), axis=1)

    lr_meta = LogisticRegression(random_state= SEED)
    lr_meta.fit(x_train,Y_train)
    pred =  lr_meta.predict(x_test)
    
    print("report", classification_report(Y_test, pred))

     ## pickle model
     # save the model to disk
    
    # pickle.dump(lr, open(r'./models/lr_first_level.sav', 'wb'))
    # pickle.dump(dtc, open(r'./models/dtc.sav', 'wb'))
    # pickle.dump(rfc, open(r'./models/rfc.sav', 'wb'))
    # pickle.dump(gau_nb, open(r'./models/gau_nb.sav', 'wb'))
    # pickle.dump(ber_nb, open(r'./models/ber_nb.sav', 'wb'))

    # pickle.dump(lr_meta, open(r'./models/lr_second_level.sav', 'wb'))
    # print("SAVE DONE")