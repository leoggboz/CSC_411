from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features

def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        for j in range(X.shape[0]):
            plt.plot(y[j], X[j][i], "bo",markersize=1.5)
            plt.xlabel(features[i])

    plt.tight_layout()
    plt.show()

def visualize_w(X, y, features,w):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        for j in range(X.shape[0]):
            plt.plot(y[j],X[j][i],  "bo",markersize=1.5)
            xlabel = features[i] + " w =" + str(w[i])
            plt.xlabel(xlabel)
    plt.tight_layout()
    plt.show()

def split_date(X,y):
    train_set_index = np.random.choice(506, 406, replace=False)
    test_set_index = list(range(506))
    for i in train_set_index:
        test_set_index.remove(i)
    train_set_X = np.array([X[i] for i in train_set_index])
    train_set_y = np.array([y[i] for i in train_set_index])
    test_set_X = np.array([X[i] for i in test_set_index])
    test_set_y = np.array([y[i] for i in test_set_index])
    return train_set_X,train_set_y,test_set_X,test_set_y

def fit_regression(X,Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    bias_x = np.ones(X.shape[0]).reshape(X.shape[0], 1)
    X_biased = np.concatenate((bias_x,X), 1)
    w_star = np.linalg.solve(np.matmul(np.transpose(X_biased),X_biased), np.matmul(np.transpose(X_biased),Y))
    return w_star

def mean_square_error(test_sample,trained_w,sample_Y):
    predicted_y = np.matmul(trained_w,test_sample.transpose())
    return np.mean((predicted_y - sample_Y) **2)

def root_mean_square_error(test_sample,trained_w,sample_Y):
    mse = mean_square_error(test_sample,trained_w,sample_Y)
    return np.sqrt(mse)

def mean_absolute_error(test_sample,trained_w,sample_Y):
    predicted_y = np.matmul(trained_w,test_sample.transpose())
    sum = 0.0
    for i in range(14):
        sum += abs(predicted_y[i] - sample_Y[i])
    return sum / 14.0

def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    # Visualize the features
    # visualize(X, y, features)

    #TODO: Split data into train and test
    train_set_X,train_set_y,test_set_X,test_set_y = split_date(X,y)
    # Fit regression model
    w = fit_regression(train_set_X, train_set_y)[1:]
    print(len(w))
    print("=====================================================")
    print("linear regression on whole set:", w)
    visualize_w(X, y, features,w)

    print("=====================================================")
    print("The trained weight w is: ",w)
    print("=====================================================")
    print("MSE of the linear regression is:",mean_square_error(test_set_X,w,test_set_y))
    print("=====================================================")
    print("Mean abolute error of the linear regression is:",mean_absolute_error(test_set_X,w,test_set_y))
    print("=====================================================")
    print("Root mean square error of the linear regression is:",root_mean_square_error(test_set_X,w,test_set_y))
    print("=====================================================")
    # featreu selection
    # normalize the data from the w to determine the most important one
    std = np.std(X, axis=0)
    mean = np.mean(X, axis=0)
    x_normalized = (X - mean)/std

    w_normalized = fit_regression(x_normalized, y)[1:]
    print("linear regression on whole normalized set:", w_normalized)
    visualize_w(x_normalized, y, features,w_normalized)

if __name__ == "__main__":
    main()
