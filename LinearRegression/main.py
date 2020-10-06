import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

'''
ECSE 551 - Mini Project 1 - Logistic Regression

This code contains the implementation of the logistic regression algorithm as a python class. 
The functions fit() and predict() are as those described in the assignment. The solution was based
on the in class notes, as well as this online tutorial:

https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html

The implementation of gradient descent and cross-entropy loss calculation differed from that shown in class
only in that it was vectorized as demonstrated in the above tutorial. 

Written by VG 2020
'''


class LogisticRegression:

    def __init__(self, features, labels, weights, learning_rate, max_iterations):
        self.x = features
        self.y = labels
        self.w = weights
        self.lr = learning_rate
        self.k = max_iterations

    def fit(self):
        # this function trains the model via gradient descent, and returns optimal weights
        # and the value of the cross entropy loss as a function of iteration number.
        # the function is broken down into several subroutines with self explanatory operations.
        def sigmoid(arg):
            return 1.0 / (1.0 + np.exp(-arg))

        def prediction(features, weights):
            return sigmoid(np.dot(features, weights))

        def update_weights(features, labels, weights, lr):
            n_obs = len(features)
            predictions = prediction(features, weights)
            gradient = np.dot(features.T, predictions - labels)
            weights = weights - lr * (gradient / n_obs)
            return weights

        def compute_loss(features, labels, weights):
            n_obs = len(features)
            predictions = prediction(features, weights)
            # Take the error when label=1
            class1_loss = -labels * np.log(predictions)
            # Take the error when label=0
            class2_loss = (1 - labels) * np.log(1 - predictions)
            # Take the sum of both costs
            loss = class1_loss - class2_loss
            # Take the average cost
            loss = loss.sum() / n_obs
            return loss

        # initiate gradient descent and track loss as a function of iteration number k.
        loss_history = np.zeros([self.k, 1])
        for i in range(self.k):
            self.w = update_weights(self.x, self.y, self.w, self.lr)
            loss_history[i] = compute_loss(self.x, self.y, self.w)

        # the fit() subroutine returns the optimal model weights and the loss function trajectory.
        return self.w, loss_history

    def predict(self):
        # this function simply evaluates the probability that a given observation falls into a given class
        # and returns the appropriate label.
        def sigmoid(arg):
            return 1.0 / (1.0 + np.exp(-arg))

        def prediction(features, weights):
            return sigmoid(np.dot(features, weights))

        def decision_boundary(probability):
            return 1 if probability >= .5 else 0

        # classify input data based on previously trained weights and return the decision.
        predictions = prediction(self.x, self.w)
        decision_boundary = np.vectorize(decision_boundary)
        return decision_boundary(predictions).flatten()


'''
Now that all class objects and methods have been defined, load some data and train the model.
'''

# read data as a data frame and convert to numpy arrays
hepatitis_data = pd.read_csv('hepatitis.csv')
numerical_data = hepatitis_data.to_numpy()

# the last column contains the true class labels
class_labels = numerical_data[:, -1]

# the rest is the feature data
'''
# any feature engineering steps can go here (normalization, feature selection, etc)
# here, I decided to cheat and use sklearn, as per this thread on stackoverflow:
# https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
# this is because feature scaling is extremely important for logistic regression to proceed well.
# from a numerical perspective, this is because the sigmoid function diverges for un-scaled feature values
# leading to the evaluation of log(1 - 1) in the cost function.
# https://stackoverflow.com/questions/36229340/divide-by-zero-encountered-in-log-when-not-dividing-by-zero/36229376
'''
input_data = numerical_data[:, 0:-1]
min_max_scale = preprocessing.MinMaxScaler()
input_data = min_max_scale.fit_transform(input_data)

# append a column of 1s to the training data for the bias term in the weights vector
n_observations = len(input_data)
input_data = np.hstack((np.ones((n_observations, 1)), input_data))

# now select subset of data to train on - let's take the first 100 for training
training_labels = class_labels[0:100]
training_data = input_data[0:100, :]

# initialize weights as a zero vector, size m+1 x 1
weights_init = np.zeros(np.size(training_data, 1))
learning_rate = 1
k_iterations = 100
# weights_init = np.random.uniform(-1, 1, np.size(training_data, 1))
training_run = LogisticRegression(training_data, training_labels,
                                  weights_init, learning_rate, k_iterations)
[model_weights, cost_trajectory] = training_run.fit()

testing_labels = class_labels[100:len(class_labels)]
testing_data = input_data[100:len(class_labels), :]
testing_run = LogisticRegression(testing_data, testing_labels,
                                 model_weights, learning_rate, k_iterations)
classified_data = testing_run.predict()

#######
plt.plot(cost_trajectory)
plt.xlabel('Iteration Number')
plt.ylabel('L(D)')
plt.ylim([0, 1])
plt.title('Loss Function History')
plt.show()

# Now we need to implement model evaluation and k-fold cross validation.
# feature expansions? log of some? add regularization as per the tutorial?