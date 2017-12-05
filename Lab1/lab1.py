#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    cancer = datasets.load_breast_cancer()
    X = cancer.data[:, :2]  # mean radius
    print(f'Take two features: {cancer.feature_names[:2]}')
    y = cancer.target

    # I.
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    # Plot the training points
    plt.figure(2, figsize=(8, 6))
    plt.clf()

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel(cancer.feature_names[0])
    plt.ylabel(cancer.feature_names[1])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    plt.show()

    X = cancer.data

    # To getter a better understanding of interaction of the dimensions
    # plot the first three PCA dimensions
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
               cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()

    for k in range(1, len(X.transpose()) + 1):
        pca = PCA(n_components=k)
        X_reduced = pca.fit_transform(X)

        cov_X = np.cov(X.transpose())
        cov_X_reduced = np.cov(X_reduced.transpose())

        trace_X = sum(cov_X[i][i] for i in range(len(cov_X)))
        if k > 1:
            trace_X_reduced = sum(cov_X_reduced[i][i] for i in range(len(cov_X_reduced)))
        else:
            trace_X_reduced = cov_X_reduced
        print(k, trace_X_reduced / trace_X)


    # II.
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.1, random_state=13)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(y_test, y_pred, target_names=target_names))

    for k_neighbors, n_folds in itertools.product(range(2, 15), [2, 5, 8, 10]):
        neigh = KNeighborsClassifier(n_neighbors=k_neighbors)
        neigh.fit(X_train, y_train)
        scores = cross_val_score(neigh, cancer.data, cancer.target, cv=n_folds)
        print("[k=%d,n=%d] Accuracy: %0.2f (+/- %0.2f)" % (k_neighbors, n_folds, scores.mean(), scores.std() * 2))

    print('\n' + '-' * 80 + '\n')

    for n_pca_components in range(2, 6):
        pca = PCA(n_components=n_pca_components)
        X_reduced = pca.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_reduced, cancer.target, test_size=0.1, random_state=13)

        for k_neighbors in range(2, 6):
            neigh = KNeighborsClassifier(n_neighbors=k_neighbors)
            neigh.fit(X_train, y_train)
            scores = cross_val_score(neigh, cancer.data, cancer.target, cv=10)
            print("[pca=%d,k=%d] Accuracy: %0.2f (+/- %0.2f)" % (n_pca_components, k_neighbors, scores.mean(), scores.std() * 2))

    print(cancer)

