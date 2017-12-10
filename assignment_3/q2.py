import numpy as np

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''

    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum

        lr - learning rate
        beta - momentum hyperparameter
    '''

    def __init__(self, lr, beta):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        self.vel = -self.lr * grad(params) + self.beta * self.vel
        return params + self.vel


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count, bias):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
        self.bias = bias
        self.feature_count = feature_count

    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        h_loss=[]
        for i in range(len(y)):
            h_loss.append(max((1 - y[i] * (np.dot(self.w, X[i]) + self.bias)),0))
        return np.array(h_loss)

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        h_loss = self.hinge_loss(X,y)
        sum_over = np.zeros(self.feature_count)
        for i in range(len(y)):
            if h_loss[i] > 0:
                sum_over += X[i] * y[i]
        return (self.w - sum_over)*(self.c / len(y))

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        classifies = []
        for i in range(X.shape[0]):
            sign = np.dot(self.w,X[i]) + self.bias
            classifies.append(sign)
        return np.sign(np.array(classifies))

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]

    for _ in range(steps):
        w = optimizer.update_params(w, func_grad)
        w_history.append(w)
    return w_history

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters, beta):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'
    '''
    batch_sampler = BatchSampler(train_data, train_targets, batchsize)
    svm = SVM(penalty, train_data[0].shape[0], 1)
    theta = 0
    for  i in range(iters):
        X_b, y_b = batch_sampler.get_batch()
        theta = -0.05 * svm.grad(X_b, y_b) + beta * theta
        svm.w += theta
    return svm

def q2_1():
    t = list(range(0, 201))
    w_history = []
    for i in range(2):
        momentumGD_optimizer = GDOptimizer(1.0,0.9*i)
        w_history.append(optimize_test_function(momentumGD_optimizer))
    plt.plot(t,w_history[0],'r')
    plt.plot(t,w_history[1],'g')
    plt.show()

def q2_2(beta):
    train_data, train_targets, test_data, test_targets = load_data()
    momentumGD_optimizer = GDOptimizer(1.0,0.9)
    # for i in range(2):
    svm = optimize_svm(train_data, train_targets, 1, momentumGD_optimizer, 100, 500, beta)

    train_accuracy = svm.classify(train_data)
    test_accuracy = svm.classify(test_data)
    # print accuracy
    print("The training set accuracy is:",(train_targets == train_accuracy).mean())
    print("The test set accuracy is:", (test_targets == test_accuracy).mean())
    print("The hinge loss of training set is :", (svm.hinge_loss(train_data,train_targets)).mean())
    print("The hinge loss of test set is :", (svm.hinge_loss(test_data,test_targets)).mean())
    plt.imshow(svm.w.reshape(28,28),cmap='gray')
    plt.show()


if __name__ == '__main__':
    # q2_1()
    q2_2(0)
    # q2_2(0.1)
    pass
