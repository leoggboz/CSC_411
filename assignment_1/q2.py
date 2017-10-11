# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.misc import logsumexp
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(1,d),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions-y_test)**2).mean()
    return losses

#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    numerator = (np.exp(-l2(test_datum, x_train)/(2*(tau**2))))[0]
    denominator = np.exp(logsumexp(-l2(test_datum, x_train)/(2*(tau**2))))
    a_weight = np.diag(np.divide(numerator , denominator))
    #w*
    left_mat = np.matmul(np.matmul(x_train.transpose(), a_weight), x_train) + lam*np.identity(d)
    right_mat = np.matmul(np.matmul(x_train.transpose(), a_weight), y_train)
    w_star = np.linalg.solve(left_mat, right_mat)
    ## TODO
    y_hat = np.matmul(test_datum, w_star)
    return y_hat[0]

def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    ## TODO
    #split data to k pieces
    shuffled_indice = np.arange(N)
    np.random.shuffle(shuffled_indice)
    loss_matrix_on_taus = []
    test_set_index = []
    for i in range(k):
        test_set_index = shuffled_indice[int(i*(N/k)): int((i+1)*(N/k))]
        train_set_index = list(range(N))
        for indice_left in test_set_index:
            train_set_index.remove(indice_left)
        x_train = np.array([x[i] for i in train_set_index])
        y_train = np.array([y[i] for i in train_set_index])
        x_test = np.array([x[i] for i in test_set_index])
        y_test = np.array([y[i] for i in test_set_index])
        loss_matrix_on_taus.append(run_on_fold(x_test, y_test, x_train, y_train, taus))
    ## TODO
    #return a vector of k-fold cross validation losses one for each tau value
    print
    return np.mean(loss_matrix_on_taus, axis=0)

if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    losses = run_k_fold(x,y,taus,k=5)
    plt.plot(losses)
    plt.show()
    print("min loss = {}".format(losses.min()))
