#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
# Based on http://scikit-learn.org/stable/auto_examples/neural_networks/plot_rbm_logistic_classification.html
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

from helpers import get_labeled_data


def main():
    classify_sklearn_8x8_digits()
    classify_big_alphabet()
    classify_mnist()


def classify_sklearn_8x8_digits():
    digits = datasets.load_digits()
    X = np.asarray(digits.data, 'float32')
    X, Y = nudge_dataset(X, digits.target)
    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    classify_logistic_rbm(X_train, Y_train, X_test, Y_test)


def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


def classify_big_alphabet():
    path = 'data/BigAlphabet_29x29_16pt_Arial_Reg'
    x, y = [], []

    for i in range(26):
        x.append(read_image_as_array(f'{path}/class-{i}.bmp'))
        y.append(i)
        for j in range(9):
            x.append(read_image_as_array(f'{path}/mutant-{i}-{j}-0.bmp'))
            y.append(i)

    X = np.asarray(x)
    Y = np.asarray(y, dtype=np.int32)

    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    classify_logistic_rbm(X_train, Y_train, X_test, Y_test)


def read_image_as_array(filename):
    # Convert to greyscale, make it flat, limit colors in [0; 15]
    matrix = np.asarray(Image.open(filename).convert('L'), dtype=np.float32)
    return np.dot(np.ravel(matrix), 15./255)


def classify_mnist():
    print("Get test set")
    testing = get_labeled_data('data/MNIST/t10k-images-idx3-ubyte.gz',
                               'data/MNIST/t10k-labels-idx1-ubyte.gz',
                               'data/MNIST/testing')
    print("Got %i testing datasets." % len(testing['x']))
    print("Get training set")
    training = get_labeled_data('data/MNIST/train-images-idx3-ubyte.gz',
                                'data/MNIST/train-labels-idx1-ubyte.gz',
                                'data/MNIST/training')
    print("Got %i training datasets." % len(training['x']))
    # Transform & trim dataset
    shape = training["x"].shape
    X_train, Y_train = training["x"].reshape(shape[0], shape[1] * shape[2]), np.ravel(training["y"])
    shape = testing["x"].shape
    X_test, Y_test = testing["x"].reshape(shape[0], shape[1] * shape[2]), np.ravel(testing["y"])
    count_train, count_test = 600, 100
    classify_logistic_rbm(X_train[:count_train, :], Y_train[:count_train],
                          X_test[:count_test, :], Y_test[:count_test])


def classify_logistic_rbm(X_train, Y_train, X_test, Y_test):
    # Training RBM-Logistic Pipeline
    logistic = linear_model.LogisticRegression(
        C=6000.0
    )
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm = BernoulliRBM(
        n_components=100,
        learning_rate=0.06,
        n_iter=20,
        random_state=0,
        verbose=True
    )
    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    classifier.fit(X_train, Y_train)

    # Training Logistic regression
    logistic_classifier = linear_model.LogisticRegression(C=100.0)
    logistic_classifier.fit(X_train, Y_train)

    # Evaluation
    print("\nLogistic regression using RBM features:\n%s\n" % (
        metrics.classification_report(
            Y_test,
            classifier.predict(X_test))))
    print("Logistic regression using raw pixel features:\n%s\n" % (
        metrics.classification_report(
            Y_test,
            logistic_classifier.predict(X_test))))

    # Plotting
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(rbm.components_):
        if i >= 100:
            break
        plt.subplot(10, 10, i + 1)
        square = int((comp.shape[0]) ** (1/2))
        plt.imshow(comp.reshape((square, square)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('100 components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    plt.show()


if __name__ == '__main__':
    main()
