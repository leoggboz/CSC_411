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

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        theta_t_1, delta_t_1 = params

        delta_t = -self.lr * grad(theta_t_1) + self.beta * delta_t_1
        theta_t = theta_t_1 + delta_t
        return theta_t, delta_t


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count, b = 0):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
        self.b = b

    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        # h_loss = np.matmul(y,np.matmul(self.w,X))
        # h_loss[h_loss<=0] = 0
        # from sklearn.metrics import hinge_loss
        # return h_loss

        h_loss=[]

        for i in range(len(y)):
            temp = (np.dot(self.w, X[i]) +self.b)
            a = 1 - y[i] * temp
            h_loss.append(max(a,0))
        return np.array(h_loss)


    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        # h_loss = self.hinge_loss(X,y)
        # gradient_hinge_loss = np.matmul(y,X)
        # print("gradient_hinge_loss:",gradient_hinge_loss.shape)
        # gradient_hinge_loss[gradient_hinge_loss <= 0] = 0
        # return gradient_hinge_loss

        to_sum = np.zeros(X.shape[1])

        h_loss = self.hinge_loss(X, y)

        for i in range(len(y)):
            if h_loss[i] <= 0:
                to_sum+= np.zeros(X.shape[1])
            else:
                to_sum += -y[i]*X[i]
        to_sum *= self.c / len(y)
        return np.array(to_sum)

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        # classifies = []
        #
        # for i in range(X.shape[0]):
        #     sign = np.dot(self.w,X[i]) + self.b
        #     classifies.append(sign)
        # return np.sign(np.array(classifies))
        classifies = []

        for i in range(X.shape[0]):
            sign  = np.dot(self.w,X[i]) + self.b
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
    delta = 0

    for _ in range(steps):
        # Optimize and update the history
        w, delta = optimizer.update_params((w,delta), func_grad)
        w_history.append(w)
    return w_history

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'
    '''
    return None


def q2_1(beta):
    Optimizer = GDOptimizer(1.0,beta)
    w_history = optimize_test_function(Optimizer)
    # now, plot.
    plt.plot(w_history)
    plt.show()


def q2_2(train_data, train_targets, test_data, test_targets, c_value, m, alpha, beta, T):
    '''
    :param train_data:
    :param train_targets:
    :param test_data:
    :param test_targets:
    :param c_value:
    :param m: mini batch size
    :param alpha: learning rate lr
    :param beta: momentum
    :param T: num of SGD iterations
    :return:
    '''


    svm = SVM(c_value, train_data[0].shape[0])
    optimizer = GDOptimizer(alpha, beta)
    delta = np.zeros(train_data.shape[1])
    delta_t_1 = np.zeros(train_data.shape[1])
    for i in range(T):
        delta_t_1 = -alpha * svm.grad(train_data, train_targets) + beta * delta_t_1
        svm.w = svm.w + delta_t_1
    #now, svm.w shall be succesfully trained.
    Train_classified = svm.classify(train_data)
    Test_classified = svm.classify(test_data)
    # print(classified)
    print("when beta =",beta,":")
    print("The classification accuracy on the training set is:",(train_targets==Train_classified).mean())
    print("The classification accuracy on the test set is:", (test_targets == Test_classified).mean())
    w_to_plot = svm.w
    w_to_plot.resize(28,28)
    plt.imshow(w_to_plot,cmap='gray')
    plt.show()




if __name__ == '__main__':
    # q2_1(0.0)
    # q2_1(0.9)
    train_data, train_targets, test_data, test_targets = load_data()

    # print("train_targets:",train_targets)
    # print("train_data:",train_data.shape)
    # print("train_targets:",train_targets.shape)
    # print(train_data[0].shape[0])
    # print(svm.grad(train_data,train_targets))

    c_value = 1.0
    m = 100
    T = 500
    alpha = 0.05
    beta = 0.8

    q2_2(train_data, train_targets, test_data, test_targets, c_value, m, alpha, beta, T)
    beta = 0.95
    q2_2(train_data, train_targets, test_data, test_targets, c_value, m, alpha, beta, T)
pass
