#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


def main():
    cancer = datasets.load_breast_cancer()
    step1(cancer)
    step2(cancer)
    step3(cancer)


def step1(cancer):
    plot_two_features(cancer)
    plot_pca(cancer)

    # Total variation for k-PCA challenge
    X = cancer.data
    cov_X = np.cov(X.transpose())
    trace_X = sum(cov_X[i][i] for i in range(len(cov_X)))
    X_eigenvalues, X_eigenvectors = np.linalg.eig(cov_X)

    column2 = "trace(X_reduced) / trace(X)"
    column3 = "Eigenvalues of cov(X)"
    column4 = "Variance of cov(X_reduced)"
    print("k", "\t", column2, "\t", column3, "\t", column4)

    for k in range(1, len(X.transpose()) + 1):
        pca = PCA(n_components=k)
        X_reduced = pca.fit_transform(X)
        cov_X_reduced = np.cov(X_reduced.transpose())

        if k > 1:
            diagonal_X_reduced = [cov_X_reduced[i][i] for i in range(len(cov_X_reduced))]
        else:
            diagonal_X_reduced = [cov_X_reduced]
        print(k, "\t",
              str(sum(diagonal_X_reduced) / trace_X).ljust(len(column2)), "\t",
              print_shorten_array(X_eigenvalues[:k], "%.2f", 50), "\t",
              print_shorten_array(diagonal_X_reduced, "%.2f", 50))
    print('=' * 160 + '\n')


def print_shorten_array(array, fmt, width):
    return textwrap.shorten(str([fmt % x for x in array]), width=width)


def plot_two_features(cancer):
    X = cancer.data[:, :2]  # mean radius
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


def plot_pca(cancer):
    # To get a better understanding of interaction of the dimensions
    # plot the first three PCA dimensions
    X = cancer.data
    y = cancer.target
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

    orthogonal_str = """Ensuring principal components are orthogonal:
    (PC[0], PC[1]) = {0:.10f}
    (PC[1], PC[2]) = {1:.10f}
    (PC[0], PC[2]) = {2:.10f}\n"""
    print(orthogonal_str.format(
        np.dot(X_reduced[:, 0], X_reduced[:, 1]),
        np.dot(X_reduced[:, 1], X_reduced[:, 2]),
        np.dot(X_reduced[:, 0], X_reduced[:, 2])
    ) + "=" * 160 + "\n\n")


def step2(cancer):
    X, y = cancer.data, cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    print("Classification report for original data. Split 90/10, 3 neighbors")
    print(classification_report(y_test, y_pred, target_names=['class 0', 'class 1']))
    plot_n_fold_vs_k_neighbors(X, y, "Original Data")
    print('\n' + '-' * 160 + '\n')


def step3(cancer):
    X, y = cancer.data, cancer.target
    for n_pca_components in range(2, 6):
        pca = PCA(n_components=n_pca_components)
        X_reduced = pca.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.1, random_state=13)
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X_train, y_train)
        y_pred = neigh.predict(X_test)

        print("Classification report for PCA (%d components). Split 90/10, 3 neighbors" % n_pca_components)
        print(classification_report(y_test, y_pred, target_names=['class 0', 'class 1']))

        plot_n_fold_vs_k_neighbors(X_reduced, y, "PCA %d components" % n_pca_components)
    print('\n' + '-' * 160 + '\n')


def plot_n_fold_vs_k_neighbors(X, y, title_suffix: str):
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    plt.title("n-fold Cross-Validation for k Neighbors " + title_suffix)
    plt.xlabel("K neighbours")
    plt.ylabel("Cross-Validation Score")

    global_min_score = 100500
    global_max_score = -100500
    k_neighbors_range = range(2, 15)

    for n_folds, color in [(2, 'r'), (5, 'g'), (8, 'b'), (10, 'c')]:
        scores_list = []
        for k_neighbors in k_neighbors_range:
            neigh = KNeighborsClassifier(n_neighbors=k_neighbors)
            scores = cross_val_score(neigh, X, y, cv=n_folds)
            scores_list.append(scores)
            # print("[k=%d,n=%d] Accuracy: %0.2f (+/- %0.2f)" % (k_neighbors, n_folds, scores.mean(), scores.std() * 2))
        min_scores = [x.min() for x in scores_list]
        max_scores = [x.max() for x in scores_list]
        global_min_score = min(global_min_score, min(min_scores))
        global_max_score = max(global_max_score, max(max_scores))

        plt.fill_between(k_neighbors_range, min_scores, max_scores, color=color, alpha=0.3)
        plt.scatter(k_neighbors_range,
                    [x.mean() for x in scores_list],
                    c=color, edgecolor='k',
                    label=str(n_folds) + '-folds')

    plt.xlim(min(k_neighbors_range), max(k_neighbors_range))
    plt.ylim(global_min_score - 0.01, global_max_score + 0.01)
    plt.legend()  # handles=[line_up, line_down])
    plt.show()


if __name__ == '__main__':
    main()

