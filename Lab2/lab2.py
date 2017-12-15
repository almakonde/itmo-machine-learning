#!/usr/bin/env python3.6

from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import nltk
import time
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import numpy as np
from sklearn import datasets, metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, CountVectorizer

VECTORIZER = 'COUNT'  # 'HASH' or 'TFIDF'
N_FEATURES = 2 ** 16  # For hashing


def clean(text: str, stop: Set[str]) -> str:
    return ' '.join([w for w in wordpunct_tokenize(text.lower()) if w.lower() not in stop])


def get_data(filename: str, cleaned: bool) -> List[List]:
    nltk.download('stopwords')
    stop = set(stopwords.words('english'))
    stop.update([c for c in '.,"\'?!:;()[]{}'] + ['\x92'])

    data = []
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            if not line.strip():
                continue

            text, label = line.strip().rsplit(',', 1)
            if label not in {'spam', 'ham'}:
                raise ValueError('Unknown label: ' + label)

            data.append([clean(text, stop) if cleaned else text, label])
    return data


def get_spam_dataset(filename: str, cleaned: bool = True) -> datasets.base.Bunch:
    x = np.array(get_data(filename, cleaned))
    idx_to_name = list(np.unique(x[:, 1]))
    name_to_idx = {name: idx for idx, name in enumerate(idx_to_name)}
    return datasets.base.Bunch(
        data=np.array(x[:, 0], dtype=np.dtype('<U16')),
        target=np.array(list(map(lambda name: name_to_idx[name], x[:, 1])),
                        dtype=np.dtype('<i8')),
        target_names=np.array(idx_to_name, dtype=np.dtype('<U16')))


def benchmark(classifier, X_train, X_test, y_train, y_test):
    if type(classifier) is GaussianNB:
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    print('_' * 80)
    print("Training: ")
    print(classifier)
    t0 = time.time()
    classifier.fit(X_train, y_train)
    train_time = time.time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time.time()
    pred = classifier.predict(X_test)
    test_time = time.time() - t0
    print("test time:  %0.3fs" % test_time)

    for n_folds in [2, 3, 4, 5]:
        scores = cross_val_score(classifier, X_train, y_train, cv=n_folds)
        print("Cross validation [n_folds=%d] Accuracy: %0.2f (+/- %0.2f)" % (
            n_folds, scores.mean(), scores.std() * 2
        ))

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    # #PRINT
    if hasattr(classifier, 'coef_'):
        print("dimensionality: %d" % classifier.coef_.shape[1])
        print("density: %f" % density(classifier.coef_))

        if feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                if i > len(classifier.coef_) - 1:
                    continue
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (label, " ".join(np.array(feature_names)[top10])))
        print()

    print("classification report:")
    print(metrics.classification_report(y_test, pred, target_names=target_names))

    # print("confusion matrix:")
    # print(metrics.confusion_matrix(y_test, pred))
    # print()

    clf_descr = str(classifier).split('(')[0]
    return clf_descr, score, train_time, test_time


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


def plot_results(results):
    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]
    clf_names, score, training_time, test_time = results

    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.barh(indices + .3, training_time, .2, label="training time",
             color='c')
    plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-0.3, i, c)

    names = enumerate(clf_names)
    for idx, method in enumerate(sorted(names, key=lambda x: -score[x[0]])):
        print('{0})\t{1}\t{2}'.format(idx + 1, method[1], score[method[0]]))

    plt.show()


def get_news_data():
    ALL_CATEGORIES = True

    if ALL_CATEGORIES:
        categories = None
    else:
        categories = [
            'alt.atheism',
            'talk.religion.misc',
            'comp.graphics',
            'sci.space',
        ]

    if True:
        remove = ('headers', 'footers', 'quotes')
    else:
        remove = ()

    print("Loading 20 newsgroups dataset for categories:")
    print(categories if categories else "all")

    data_train = fetch_20newsgroups(subset='train', categories=categories,
                                    shuffle=True, random_state=42,
                                    remove=remove)

    data_test = fetch_20newsgroups(subset='test', categories=categories,
                                   shuffle=True, random_state=42,
                                   remove=remove)
    return data_train, data_test, categories


# data_train = get_dataset('corpus/english_big.txt')
# data_test = get_dataset('corpus/english.txt', cleaned=False)
data_train = get_spam_dataset('D:/ML/Lab2/corpus/english_big.txt')
data_test = get_spam_dataset('D:/ML/Lab2/corpus/english.txt', cleaned=False)
categories = ['spam', 'ham']
# data_train, data_test, categories = get_news_data()
print('data loaded')

# order of labels in `target_names` can be different from `categories`
target_names = data_train.target_names

data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)

print("%d documents - %0.3fMB (training set)" % (len(data_train.data), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (len(data_test.data), data_test_size_mb))
print("{0} categories".format(len(categories) if categories else 'all'))
print()

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training data using a sparse vectorizer")
t0 = time.time()
if VECTORIZER == 'HASH':
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False, n_features=N_FEATURES)
    X_train = vectorizer.transform(data_train.data)
elif VECTORIZER == 'TFIDF':
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)
else:
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(data_train.data)

# Print matrix
# X_train.toarray().astype(int)
print("done in %fs" % (time.time() - t0))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer")
t0 = time.time()
X_test = vectorizer.transform(data_test.data)
print("done in %fs" % (time.time() - t0))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

# mapping from integer feature name to original token string
if VECTORIZER == 'HASH':
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()

"""
    Classifiers



















"""

results = []

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01), X_train, X_test, y_train, y_test))
results.append(benchmark(BernoulliNB(alpha=.01), X_train, X_test, y_train, y_test))
results.append(benchmark(GaussianNB(), X_train, X_test, y_train, y_test))
# for clf, name in (
#         (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
#         (Perceptron(n_iter=50), "Perceptron"),
#         (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
#         (KNeighborsClassifier(n_neighbors=10), "kNN"),
#         (RandomForestClassifier(n_estimators=100), "Random forest"),
# ):
#     print('=' * 80)
#     print(name)
#     results.append(benchmark(clf, X_train, X_test, y_train, y_test))
#
# for penalty in ["l2", "l1"]:
#     print('=' * 80)
#     print("%s penalty" % penalty.upper())
#     # Train Liblinear model
#     results.append(benchmark(LinearSVC(penalty=penalty, dual=False, tol=1e-3),
#                              X_train, X_test, y_train, y_test))
#
#     # Train SGD model
#     results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty=penalty),
#                              X_train, X_test, y_train, y_test))
#
# # Train SGD with Elastic Net penalty
# print('=' * 80)
# print("Elastic-Net penalty")
# results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"),
#                          X_train, X_test, y_train, y_test))
#
# # Train NearestCentroid without threshold
# print('=' * 80)
# print("NearestCentroid (aka Rocchio classifier)")
# results.append(benchmark(NearestCentroid(), X_train, X_test, y_train, y_test))
#
# print('=' * 80)
# print("LinearSVC with L1-based feature selection")
# # The smaller C, the stronger the regularization.
# # The more regularization, the more sparsity.
# results.append(benchmark(Pipeline([
#     ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3))),
#     ('classification', LinearSVC(penalty="l2"))]),
#     X_train, X_test, y_train, y_test))

# make some plots
plot_results(results)
