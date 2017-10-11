import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

BATCHES = 50

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

def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w

def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

#TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    batch_grad = 2*np.matmul(np.matmul(X.transpose(),X),w) - 2*np.matmul(X.transpose(),y)
    return batch_grad / X.shape[0]

def min_batch_gradient(X,y,w,m):
    batch_grad = 0
    batch_sampler = BatchSampler(X, y, m)
    batch_matrix = []
    for i in range(500):
        X_b, y_b = batch_sampler.get_batch()
        batch_grad += lin_reg_gradient(X_b, y_b, w)
        batch_matrix.append(lin_reg_gradient(X_b, y_b, w))
    batch_grad = batch_grad / m
    return batch_grad, batch_matrix

def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)

    # Example usage
    batch_grad = 0
    for i in range(500):
        X_b, y_b = batch_sampler.get_batch()
        batch_grad += lin_reg_gradient(X_b, y_b, w)
    batch_grad = batch_grad / 500
    true_grad = lin_reg_gradient(X, y, w)

    print(batch_grad)
    print(true_grad)
    print("The squared distance between mini-batch gradient and gradient of the whole set: ",(np.linalg.norm(batch_grad-true_grad))**2)
    print("The cosine similarity between mini-batch gradient and gradient of the whole set",cosine_similarity(batch_grad,true_grad))

    batch_grad_list = []
    log_m = []
    log_sigma = []
    for i in range(1,401):
        batch_grad_m,batch_matrix = min_batch_gradient(X,y,w,i)
        #calculate variance
        batch_grad_list.append( np.log(np.sum( ((batch_matrix - np.mean(batch_matrix, axis = 0))**2),axis =0 )/ 500))
        log_m.append(np.log(i))
    plt.figure(figsize=(20, 5))
    for i in range(len(X[0])):
        plt.subplot(3, 5, i + 1)
        for j in range(len(batch_grad_list)):
            plt.plot(log_m[j],batch_grad_list[j][i], "bo",markersize=1.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
