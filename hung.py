import numpy as np
import pandas as pd

from sklearn.datasets import make_regression

from sklearn.metrics import r2_score

class Regression:
    def __init__(self, learning_rate, iteration, regularization):
        """
        learning rate: tốc độ học
        iteration: số vòng lặp đào tạo
        """
        self.m = None
        self.n = None
        self.w = None
        self.b = None
        self.regularization = regularization # tham số chuẩn hóa theo mô hình hồi quy
        self.lr = learning_rate
        self.it = iteration

    def cost_function(self, y, y_pred):
        """
        y: giá trị mục tiêu.
        y_pred: giá trị dự đoán.
        """
        return (1 / (2*self.m)) * np.sum(np.square(y_pred - y)) + self.regularization(self.w)
    
    def hypothesis(self, weights, bias, X):
        """
        weights: parameter value weight.
        X: mẫu đào tạo.
        """
        return np.dot(X, weights) #+ bias

    def train(self, X, y):
        """
        X: tập dữ liệu huấn luyện ---> N Dimentional vector.
        y: giá trị mục tiêu của tập dữ liệu huấn luyện -----> 1 Dimentional array.
        """
        # Insert constant ones for bias weights.
        X = np.insert(X, 0, 1, axis=1)

        # Target value should be in the shape of (n, 1) not (n, ).
        # So, this will check that and change the shape to (n, 1), if not.
        try:
            y.shape[1]
        except IndexError as e:
            # Cần thay đổi nó thành mảng 1 D, không phải list.
            print("LỖI: Mảng mục tiêu phải là mảng một chiều không phải là danh sách" 
            "----> ở đây giá trị mục tiêu không ở dạng (n,1). \nShape ({shape_y_0},1) and {shape_y} not match"
                  .format(shape_y_0 = y.shape[0] , shape_y = y.shape))
            return 
        
        # m : số lượng mẫu đào tạo.
        self.m = X.shape[0]
        # n số các thuộc tính .
        self.n = X.shape[1]

        # Set the initial weight.
        self.w = np.zeros((self.n , 1))

        # bias.
        self.b = 0

        for it in range(1, self.it+1):
            # 1. Find the predicted value through the hypothesis.
            # 2. Find the Cost function value.
            # 3. Find the derivation of weights.
            # 4. Apply Gradient Decent.
            y_pred = self.hypothesis(self.w, self.b, X)
            #print("iteration",it)
            #print("y predict value",y_pred)
            cost = self.cost_function(y, y_pred)
            #print("Cost function",cost)
            # fin the derivative.
            dw = (1/self.m) * np.dot(X.T, (y_pred - y)) + self.regularization.derivation(self.w)
            #print("weights derivation",dw)
            #db = -(2 / self.m) * np.sum((y_pred - y))

            # change the weight parameter.
            self.w = self.w - self.lr * dw
            #print("updated weights",self.w)
            #self.b = self.b - self.lr * db


            if it % 10 == 0:
                print("The Cost function for the iteration {}----->{} :)".format(it, cost))
    def predict(self, test_X):
        """
        :param test_X: feature values to predict.
        """
        # Insert constant ones for bias weights.
        test_X = np.insert(test_X, 0, 1, axis=1)

        y_pred = self.hypothesis(self.w, self.b, test_X)
        return y_pred

class l2_regularization:
    """Regularization used for Ridge Regression"""
    def __init__(self, lamda):
        self.lamda = lamda

    def __call__(self, weights):
        "This will be retuned when we call this class."
        return self.lamda * np.sum(np.square(weights))
    
    def derivation(self, weights):
        "Derivation of the regulariozation function."
        return self.lamda * 2 * (weights)
class RidgeRegression(Regression):
    """
    Ridge Regression is one of the variance of the Linear Regression. This model doing the parameter learning 
    and regularization at the same time. This model uses the l2-regularization. 
    This is very similar to the Lasso regression.
    * Regularization will be one of the soluions to the Overfitting.
    * Overfitting happens when the model has "High Variance and low bias". So, regularization adds a little bias to the model.
    * This model will try to keep the balance between learning the parameters and the complexity of the model( tries to keep the parameter having small value and small degree of palinamial).
    * The Regularization parameter(lamda) controls how severe  the regularization is. 
    * large lamda adds more bias , hence the Variance will go very small --> this may cause underfitting(Low bias and High Varinace).
    * Lamda can be found by tial and error methos. 
    """
    def __init__(self, lamda, learning_rate, iteration):
        """
        Define the hyperparameters we are going to use in this model.
        :param lamda: Regularization factor.
        :param learning_rate: A samll value needed for gradient decent, default value id 0.1.
        :param iteration: Number of training iteration, default value is 10,000.
        """
        self.regularization = l2_regularization(lamda)
        super(RidgeRegression, self).__init__(learning_rate, iteration, self.regularization)

    def train(self, X, y):
        """
        :param X: training data feature values ---> N Dimentional vector.
        :param y: training data target value -----> 1 Dimentional array.
        """
        return super(RidgeRegression, self).train(X, y)
    def predict(self, test_X):
        """
        parma test_X: Value need to be predicted.
        """
        return super(RidgeRegression, self).predict(test_X)