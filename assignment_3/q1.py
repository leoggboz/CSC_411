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
from sklearn.naive_bayes import GaussianNB


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

    #grid search
    tol_log = [1e-2,1e-3,1e-4]
    pen_log = ['l1','l2'];
    for i in tol_log:
        for j in pen_log:
            print("Tolenrance is %f, penalty is %s" %(i ,j))
            model = LogisticRegression(tol = i,
                                       penalty=j,
                                       fit_intercept=True,
                                       max_iter=200,
                                       random_state=42,
                                       )

            model.fit(bow_train, train_labels)
            train_pred = model.predict(bow_train)
            print('LogisticRegression  train accuracy = {}'.format((train_pred == train_labels).mean()))
            test_pred = model.predict(bow_test)
            print('LogisticRegression  test accuracy = {}\n'.format((test_pred == test_labels).mean()))
    return model

def SVMClassifier(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    # binary_train = (bow_train>0).astype(int)
    # binary_test = (bow_test>0).astype(int)

    alpah_svm = [0.0001, 0.001]
    max_iter_svm = [500, 1000]
    for i in alpah_svm:
        for j in max_iter_svm:
            print("alpah_svm is %f, max_iter_svm is %d" %(i ,j))
            model = SGDClassifier(alpha=i, average=False, class_weight=None, epsilon=0.1,
               eta0=0.0, fit_intercept=True, l1_ratio=0.15,
               learning_rate='optimal', loss='hinge', max_iter=j, n_iter=None,
               n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
               shuffle=True, tol=1e-3, verbose=0, warm_start=False)
            model.fit(bow_train, train_labels)

            train_pred = model.predict(bow_train)
            print('SGDClassifier train accuracy = {}'.format((train_pred == train_labels).mean()))
            test_pred = model.predict(bow_test)
            print('SGDClassifier test accuracy = {}\n'.format((test_pred == test_labels).mean()))

    return model

def neuralNetwork(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    # binary_train = (bow_train>0).astype(int)
    # binary_test = (bow_test>0).astype(int)

    # grid search
    alpha_neural = [1e-5 , 1e-6, 1e-6]
    hidden_neural = [(15,10),(10,2),(10,5)]
    random_state_neural = [1,2,3,4]
    for i in alpha_neural:
        for j in hidden_neural:
            for m in random_state_neural:
                print("alpha is %f, hidden_layer is[%d,%d], random random_state is %d" %(i, j[0], j[1], m))
                model = MLPClassifier(solver='lbfgs', alpha=i, hidden_layer_sizes=j, random_state=m)
                model.fit(bow_train, train_labels)

                train_pred = model.predict(bow_train)
                print('Neural network train accuracy = {}'.format((train_pred == train_labels).mean()))
                test_pred = model.predict(bow_test)
                print('Neural network test accuracy = {}\n'.format((test_pred == test_labels).mean()))

    return model

def best_model(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    # binary_train = (bow_train>0).astype(int)
    # binary_test = (bow_test>0).astype(int)

    model = LogisticRegression(tol = 1e-3,
                               penalty='l1',
                               fit_intercept=True,
                               max_iter=200,
                               random_state=42,
                               )

    model.fit(bow_train, train_labels)
    train_pred = model.predict(bow_train)
    print('LogisticRegression  train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test)
    print('LogisticRegression  test accuracy = {}\n'.format((test_pred == test_labels).mean()))
    return model

def conf_matrix(test_labels, test_pred):
    CM =  np.zeros(((max(test_labels)+1),(max(test_labels))+1))
    two_label = np.zeros(((max(test_labels)+1),(max(test_labels))+1))
    for i in range(test_pred.shape[0]):
        CM[test_labels[i]][test_pred[i]] += 1
    for i in range(20):
        for j in range(20):
            two_label[i][j] = CM[i][j] + CM[j][i]

    # return CM.astype(int), two_label.astype(int)
    return CM, two_label


if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    # tf_idf_train, tf_idf_test, feature_names = tf_idf_features(train_data, test_data)

    # bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)

    # logisticReg(train_bow, train_data.target, test_bow, test_data.target)

    # SVMClassifier(train_bow, train_data.target, test_bow, test_data.target)

    # neuralNetwork(train_bow, train_data.target, test_bow, test_data.target)

    model = best_model(train_bow, train_data.target, test_bow, test_data.target)

    print("My best model is LogisticRegression and here is its confusion matrix")
    test_pred = model.predict(test_bow)
    cm, two_label = conf_matrix(test_data.target, test_pred)

    mean_confusion = np.sum(cm, axis = 1)
    for i in range(20):
        cm[i] = cm[i] / mean_confusion[i]
        two_label[i] = two_label[i] / mean_confusion[i]
    print(cm)
    print()
    print(np.round(two_label, decimals = 2))
    print("LogisticRegression are most confused about %s, %s." %(train_data.target_names[18], train_data.target_names[19]))



    for i in range(20):
        print(train_data.target_names[i])
