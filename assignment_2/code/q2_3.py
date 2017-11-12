'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    etas = []
    for i in range(10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        total = i_digits[0]
        for j in range(1, i_digits.shape[0]):
            total = np.add(total, i_digits[j])
        eta = (total + 1) / (i_digits.shape[0] + 2)
        etas.append(eta)
    return np.array(etas)

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    imgs = []
    for i in range(10):
        img_i = class_images[i]
        imgs.append(img_i.reshape((8, 8)))
    all_concat = np.concatenate(imgs, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    new_data = []
    for i in range(10):
        i_digit = []
        for j in range(64):
            i_digit.append(np.random.binomial(1, eta[i][j]))
        new_data.append(i_digit)
    generated_data = np.array(new_data)
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array
    '''
    generative_likelihood = []
    for i in range(bin_digits.shape[0]):
        i_digit = []
        i_likelihood = 1
        for d in range(eta.shape[0]):
            digit_likeihoood = 0
            for k in range(64):
                if bin_digits[i][k] == 1:
                    digit_likeihoood += np.log(eta[d][k])
                else:
                    digit_likeihoood += np.log(1 - eta[d][k])
            i_digit.append(digit_likeihoood)
        generative_likelihood.append(i_digit)
    return np.array(generative_likelihood)

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    gen_likelihood = generative_likelihood(bin_digits, eta)
    evidence = np.exp(gen_likelihood.copy())
    evidence = np.log(np.mean(evidence, axis = 1).reshape(evidence.shape[0],1))
    evidence = np.tile(evidence, 10)
    return gen_likelihood + np.log(0.1) - evidence

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute as described above and return
    true_likeihood = []
    labels.astype(int)

    for index in range(labels.shape[0]):
        temp = int(labels[index])
        true_likeihood.append(cond_likelihood[index][temp])
    true_likeihood = np.asarray(true_likeihood)
    return np.mean(true_likeihood, axis=0)

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)
    generate_new_data(eta)
    print(np.exp(generative_likelihood(train_data, eta)))
    print(np.exp(conditional_likelihood(train_data,eta)))
    print("Average conditional likelihood over the true training class labels is %f" %avg_conditional_likelihood(train_data, train_labels, eta))
    print("Average conditional likelihood over the true testing class labels is %f" %avg_conditional_likelihood(test_data, test_labels, eta))

    classifier accuracy
    classify = classify_data(test_data, eta)
    correct = 0
    for i in range(classify.shape[0]):
        if classify[i] == test_labels[i]:
            correct += 1
    accuracy = correct / classify.shape[0]
    print("Naive bayes classifier on testing set has an accuracy of %f." %accuracy)

    classify = classify_data(train_data, eta)
    correct = 0
    for i in range(classify.shape[0]):
        if classify[i] == train_labels[i]:
            correct += 1
    accuracy = correct / classify.shape[0]
    print("Naive bayes classifier on training set has an accuracy of %f." %accuracy)


if __name__ == '__main__':
    main()
