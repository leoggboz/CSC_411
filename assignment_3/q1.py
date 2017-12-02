'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier


def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def logisticReg(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    # binary_train = (bow_train>0).astype(int)
    # binary_test = (bow_test>0).astype(int)

    model = LogisticRegression()
    model.fit(bow_train, train_labels)

    train_pred = model.predict(bow_train)
    print('LogisticRegression  train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test)
    print('LogisticRegression  test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def SVMClassifier(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    # binary_train = (bow_train>0).astype(int)
    # binary_test = (bow_test>0).astype(int)

    model = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=1000, n_iter=None,
       n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
       shuffle=True, tol=1e-3, verbose=0, warm_start=False)
    model.fit(bow_train, train_labels)

    train_pred = model.predict(bow_train)
    print('SGDClassifier train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test)
    print('SGDClassifier test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def neuralNetwork(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    # binary_train = (bow_train>0).astype(int)
    # binary_test = (bow_test>0).astype(int)

    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 10), random_state=1)
    model.fit(bow_train, train_labels)

    train_pred = model.predict(bow_train)
    print('MLPClassifier train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test)
    print('MLPClassifier test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def conf_matrix(test_labels, test_pred):
    CM =  np.zeros(((max(test_labels)+1),(max(test_labels))+1))

    for i in range(test_pred.shape[0]):
        CM[test_labels[i]][test_pred[i]] += 1
    return CM.astype(int)


if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    tf_idf_train, tf_idf_test, feature_names = tf_idf_features(train_data, test_data)

    # bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)

    model = logisticReg(train_bow, train_data.target, test_bow, test_data.target)

    # SVMClassifier(train_bow, train_data.target, test_bow, test_data.target)
    #
    # neuralNetwork(train_bow, train_data.target, test_bow, test_data.target)

    print("My best model is LogisticRegression and here is its confusion matrix")
    test_pred = model.predict(test_bow)
    print(conf_matrix(test_data.target, test_pred))
