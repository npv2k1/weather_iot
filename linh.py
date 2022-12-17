import numpy as np
import pandas as pd

from sklearn.datasets import make_regression

from sklearn.metrics import r2_score

class LinearRegression:
    def __init__(self, learning_rate, iteration):
        """
        :param learning_rate: A samll value needed for gradient decent, default value id 0.1.
        :param iteration: Number of training iteration, default value is 10,000.
        """
        self.m = None
        self.n = None
        self.w = None
        self.lr = learning_rate
        self.it = iteration

    def cost_function(self, y, y_pred):
        """
        :param y: Original target value.
        :param y_pred: predicted target value.
        """
        return (1 / (2*self.m)) * np.sum(np.square(y_pred - y))
    
    def hypothesis(self, weights, X):
        """
        :param weights: parameter value weight.
        :param X: Training samples.
        """
        return np.dot(X, weights)

    def train(self, X, y):
        """
        :param X: training data feature values ---> N Dimentional vector.
        :param y: training data target value -----> 1 Dimentional array.
        """
        # Insert constant ones for bias weights.
        X = np.insert(X, 0, 1, axis=1)
        # Target value should be in the shape of (n, 1) not (n, ).
        # So, this will check that and change the shape to (n, 1), if not.
        try:
            y.shape[1]
        except IndexError as e:
            # we need to change it to the 1 D array, not a list.
            print("ERROR: Target array should be a one dimentional array not a list"
                  "----> here the target value not in the shape of (n,1). \nShape ({shape_y_0},1) and {shape_y} not match"
                  .format(shape_y_0 = y.shape[0] , shape_y = y.shape))
            return 
        
        # m is the number of training samples.
        self.m = X.shape[0]
        # n is the number of features.
        self.n = X.shape[1]

        # Set the initial weight.
        self.w = np.zeros((self.n , 1))

        for it in range(1, self.it+1):
            # 1. Find the predicted value through the hypothesis.
            # 2. Find the Cost function value.
            # 3. Find the derivation of weights.
            # 4. Apply Gradient Decent.
            y_pred = self.hypothesis(self.w, X)

            cost = self.cost_function(y, y_pred)
            # fin the derivative.
            dw = (1/self.m) * np.dot(X.T, (y_pred - y))

            # change the weight parameter.
            self.w = self.w - self.lr * dw

            if it % 1000 == 0:
                print("The Cost function for the iteration {}----->{} :)".format(it, cost))
    def predict(self, test_X):
        """
        :param test_X: feature values to predict.
        """
        # Insert constant ones for bias weights
        test_X = np.insert(test_X, 0, 1, axis=1)
        y_pred = self.hypothesis(self.w, test_X)
        return y_pred