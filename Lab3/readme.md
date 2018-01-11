## Practical assignment № 3 «Text clustering»

Modules of scikit: sklearn.feature_extraction.text, sklearn.cluster, cross_validation, metrics

1. Choose one of the datasets: posts from a community of a bank in a social network (posts_1.json) or posts and comments from community about traffic incidents (posts_comments_2.json).
2. The goal is to cluster posts (for dataset 1) or comments (for dataset 2).
3. Apply basic text preprocessing steps (the example can be found [here](https://www.kdnuggets.com/2017/06/text-clustering-unstructured-data.html).
4. Apply at least two different clustering approaches: a) hierarchical clustering;
b) k-means.
5. Create dendrograms for hierarchical clustering.
6. Provide representatives of each cluster (or cluster centers). Try to interpret the results of clustering.
7. Compare quality of clustering for different approaches. Try to find the best parameters (number of clusters, method of text representation, metric of distance) for a given problem.

**NB 1.** To represent text, you can use:

 * TF-IDF matrix;
 * [doc2vec](https://radimrehurek.com/gensim/models/doc2vec.html);
 * topics extracted with LSA or LDA (examples are [here](https://github.com/scikit-learn/scikit-learn/blob/master/examples/text/document_clustering.py), and [here](https://radimrehurek.com/gensim/wiki.html)).

**NB 2.** Some basic things can be found [here](http://stp.lingfil.uu.se/~santinim/ml/UnsupervisedLearningMagnusRosell_Slides.pdf)

## NB 3.

The task is not finished, however, you can find useful comments inside.
