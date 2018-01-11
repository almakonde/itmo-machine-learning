#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import json
import logging
import sys
from optparse import OptionParser
from time import time
from pprint import pprint
from typing import List

from matplotlib import pyplot as plt
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist
from sklearn import metrics
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans

from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster

from plots import plot_clusters_d3, plot_clusters_mpl

import sys
sys.path.append('..')
from contrib.text import clean_text
sys.path.append('Lab3')

from nltk.stem.snowball import RussianStemmer

FILENAME = 'posts_1.json'
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
stop_words.update([c for c in '.,"\'?!:;()[]{}'])
stemmer = RussianStemmer(False)

cluster_colors = {1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5: '#1b9e77', 6: "red", 7: "green"}
cluster_names = {1: 'Cluster 1',
                 2: 'Cluster 2',
                 3: 'Cluster 3',
                 4: 'Cluster 4',
                 5: 'Cluster 5',
                 6: 'Cluster 6',
                 7: 'Cluster 7'}


def main():
    corpus = get_corpus(FILENAME)
    print("%d documents\n" % len(corpus))
    #
    # vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    # X = vectorizer.fit_transform(corpus)
    #
    # inv_vectorizer = {}
    # for idx in range(len(corpus)):
    #     inv_vectorizer[hash(str(X[idx].data))] = corpus[idx]
    #
    # res = KMeans().fit(X)
    # for center in res.cluster_centers_:
    #     pprint(inv_vectorizer.get(hash(str(center))))
    # pprint(res)

    # Display progress logs on stdout
    opts = get_options()

    # labels = dataset.target
    # true_k = np.unique(labels).shape[0]
    # true_k = 7

    print("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time()
    vectorizer = get_vectorizer(opts)
    X = vectorizer.fit_transform(corpus)
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, m_features: %d\n" % X.shape)

    # if opts.n_components:
    #     print("Performing dimensionality reduction using LSA")
    #     t0 = time()
    #     # Vectorizer results are normalized, which makes KMeans behave as
    #     # spherical k-means for better results. Since LSA/SVD results are
    #     # not normalized, we have to redo the normalization.
    #     svd = TruncatedSVD(opts.n_components)
    #     normalizer = Normalizer(copy=False)
    #     lsa = make_pipeline(svd, normalizer)
    #
    #     X = lsa.fit_transform(X)
    #     print("done in %fs" % (time() - t0))
    #
    #     explained_variance = svd.explained_variance_ratio_.sum()
    #     print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
    #     print()
    # else:
    #     svd = None

    # clusters = cluster(X, opts, svd, true_k, vectorizer)
    clusters = hierarchical_clustering(X)

    terms = vectorizer.get_feature_names()
    dist = 1 - cosine_similarity(X)
    xs, ys = project_to_plane(dist)
    # create data frame that has the result of the MDS plus the cluster numbers and titles
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=[x.ljust(30) for x in corpus]))
    # plot_clusters_mpl(df, cluster_names, cluster_colors)
    plot_clusters_d3(df, cluster_names, cluster_colors)


def get_corpus(filename):
    with open(filename, encoding='utf-16') as data_file:
        posts = json.load(data_file)
    for post in posts:
        post["text"] = clean_text(post["text"], stop_words, '«»|-—')
    # pprint(posts[:10])
    corpus = [post['text'] for post in posts]
    return corpus


def get_options():
    # parse commandline arguments
    op = OptionParser()
    op.add_option("--lsa",
                  dest="n_components", type="int",
                  help="Preprocess documents with latent semantic analysis.")
    op.add_option("--no-minibatch",
                  action="store_false", dest="minibatch", default=True,
                  help="Use ordinary k-means algorithm (in batch mode).")
    op.add_option("--no-idf",
                  action="store_false", dest="use_idf", default=True,
                  help="Disable Inverse Document Frequency feature weighting.")
    op.add_option("--use-hashing",
                  action="store_true", default=False,
                  help="Use a hashing feature vectorizer")
    op.add_option("--n-features", type=int, default=10000,
                  help="Maximum number of features (dimensions)"
                       " to extract from text.")
    op.add_option("--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Print progress reports inside k-means algorithm.")
    # print(__doc__)
    # op.print_help()
    # work-around for Jupyter notebook and IPython console
    argv = [] if is_interactive() else sys.argv[1:]
    (opts, args) = op.parse_args(argv)
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)
    return opts


def tokenize_and_stem(text: str) -> List[str]:
    return [stemmer.stem(t) for t in clean_text(text).split(' ')]


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


def get_vectorizer(opts):
    if opts.use_hashing:
        if opts.use_idf:
            # Perform an IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(n_features=opts.n_features,
                                       stop_words=list(stop_words), alternate_sign=False,
                                       norm=None, binary=False)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(n_features=opts.n_features,
                                           stop_words=list(stop_words),
                                           alternate_sign=False, norm='l2',
                                           binary=False)
    else:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                     min_df=1, stop_words=list(stop_words),
                                     use_idf=True,
                                     tokenizer=tokenize_and_stem,
                                     ngram_range=(1, 3))
    return vectorizer


def cluster(X, opts, svd, true_k, vectorizer):
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)
    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    print()
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, km.labels_, sample_size=1000))
    print()
    if not opts.use_hashing:
        print("Top terms per cluster:")

        if opts.n_components:
            original_space_centroids = svd.inverse_transform(km.cluster_centers_)
            order_centroids = original_space_centroids.argsort()[:, ::-1]
        else:
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        terms = vectorizer.get_feature_names()
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :15]:
                print(' %s' % terms[ind], end='')
            print()


def hierarchical_clustering(X):
    if type(X) is csr_matrix:
        X = X.todense()

    # generate the linkage matrix
    # method (cophenet) clusters(Elbow):
        # 'single' (0.50659) 1,
        # 'complete' (0.52470) 5,
        # 'average' (0.58696) 6,
        # 'weighted' (0.53701) 7,
        # 'centroid' (0.55119) 9,
        # 'median' (0.47276) 7,
        # 'ward' (0.28674) 3,
    # 'euclidean' (default),
    # 'cityblock' aka Manhattan,
    # 'hamming',
    # 'cosine'
    Z = linkage(X, method='ward')

    c, coph_dists = cophenet(Z, pdist(X))
    print("Cophenetic correlation: %.5f" % c)

    # calculate full dendrogram
    if False:
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8,  # font size for the x axis labels
        )
        plt.show()

    if False:
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendrogram(
            Z,
            truncate_mode='lastp',  # show only the last p merged clusters
            p=12,  # show only the last p merged clusters
            show_leaf_counts=False,  # otherwise numbers in brackets are counts
            leaf_rotation=90.,
            leaf_font_size=12,
            show_contracted=True,  # to get a distribution impression in truncated branches
        )
        plt.show()

    if False:
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        dendrogram(
            Z,
            truncate_mode='lastp',  # show only the last p merged clusters
            p=12,  # show only the last p merged clusters
            leaf_rotation=90.,
            leaf_font_size=12,
            show_contracted=True,  # to get a distribution impression in truncated branches
        )
        plt.show()

    # set cut-off
    last_steps = 6
    max_d = 2.5  # max_d as in max_distance

    fancy_dendrogram(
        Z,
        truncate_mode='lastp',
        p=last_steps,
        leaf_rotation=90.,
        leaf_font_size=12,
        show_contracted=True,
        annotate_above=1,
        max_d=max_d,  # plot a horizontal cut-off line
    )
    plt.show()

    # Elbow Method
    last = Z[-last_steps:, 2]
    # last_rev = last[::-1]
    # idxs = np.arange(1, len(last) + 1)
    # plt.plot(idxs, last_rev)

    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    # plt.plot(idxs[:-2] + 1, acceleration_rev)
    # plt.show()
    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    print("clusters:", k)

    clusters = fcluster(Z, max_d, criterion='distance')
    print(clusters.max())

    clusters = fcluster(Z, k, criterion='maxclust')
    print(clusters.max())
    return clusters


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def project_to_plane(dist):
    # project to a plane using multidimensional scaling
    MDS()

    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

    return pos[:, 0], pos[:, 1]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
