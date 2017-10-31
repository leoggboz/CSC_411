'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from collections import Counter

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

        # print(self.train_norm)
        # print(self.train_norm.shape)
        # print(self.train_data)
        # print(self.train_data.shape)
        # print(self.train_labels)
        # print(self.train_labels.shape)

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point

        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        digit = []
        knn_distance = self.l2_distance(test_point)
        knn_first_k = np.sort(knn_distance,axis=None)
        for i in range(k):
            digit.append(self.train_labels[np.where( knn_distance == knn_first_k[i] )][0])
        label = Counter(digit)
        return label.most_common()[0][0]

def cross_validation(knn, k_range=np.arange(1,15)):
    for k in k_range:
        print("Use %d NN" %k)
        X = np.array(knn.train_data)
        y = np.array(knn.train_labels)
        kf = KFold(n_splits=10)
        kf.get_n_splits(X)
        kfold_accuracy = 0.0
        for train_index, test_index in kf.split(X):
            temp_knn = KNearestNeighbor(X[train_index], y[train_index])
            kfold_accuracy += classification_accuracy(temp_knn,k,X[test_index],y[test_index])
            print(" inside loop %f"%kfold_accuracy)
        print(" just came out of loop %f" %kfold_accuracy)
        kfold_accuracy = kfold_accuracy/10
        print(" accuracy is %f\n" %kfold_accuracy)

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    accuracy_k = 0
    for i in range(eval_data.shape[0]):
        if( knn.query_knn(eval_data[i], k) == eval_labels[i] ):
            accuracy_k += 1

    return accuracy_k/eval_data.shape[0]

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)
    # Example usage:
    # print(classification_accuracy(knn,1,test_data,test_labels))
    # print(classification_accuracy(knn,15,test_data,test_labels))
    cross_validation(knn)


if __name__ == '__main__':
    main()
