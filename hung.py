import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

class Regression:
    '''BƯỚC 1: Tạo các tham số cho mô hình'''
    def __init__(self, learning_rate, iteration, regularization):
        self.m = None #số mẫu đào tạo
        self.n = None # số thuộc tính 
        self.w = None #trọng số
        self.b = None # gía trị bias
        self.regularization = regularization # phần chính quy hóa giúp giảm bớt phức tạp, giúp giảm hiện tượng quá khớp(overfiting)
        self.lr = learning_rate #gtrị tốc độ học
        self.it = iteration # số lần lặp đào tạo

    '''BƯỚC 2: Xây dựng hàm theo dõi lỗi toàn bộ tập dữ liệu, trả về giá trị càng nhỏ thì độ khớp của tập dliệu đào tạo và dự đoán cao'''
    def cost_function(self, y, y_pred):
        return (1 / (2*self.m)) * np.sum(np.square(y_pred - y)) + self.regularization(self.w)
    
    '''BƯỚC 3: Xây dựng hàm giả thuyết phương trình y = bias + X1.w1+....'''
    def hypothesis(self, weights, bias, X): 
        # weights: giá trị trọng số
        # X: các mẫu đào tạo
        return np.dot(X, weights) + bias
    '''BƯỚC 4: Xây dựng hàm đào tạo và dự đoán'''
    def train(self, X, y):
        # X: tập dữ liệu huấn luyện là vector N chiều.
        # y: giá trị mục tiêu của tập dữ liệu huấn luyện là mảng 1 chiều.

        X = np.insert(X, 0, 1, axis=1) #chèn cột gtrị bias=1 vào đầu X
        # mảng y phải có dạng (n,1), nếu kphải in ra lỗi và chạy tiếp
        try:
            y.shape[1]
        except IndexError as e:
            # Cần thay đổi nó thành mảng 1 D, không phải list.
            print("LỖI: Mảng mục tiêu phải không là mảng 1D")
            return 
        
        # m : số lượng mẫu đào tạo.
        self.m = X.shape[0]
        # n số các thuộc tính .
        self.n = X.shape[1]
        # gán giá trị trọng số ban đầu: w là 1 ma trận 2 chiều kích thước [n,1]
        self.w = np.zeros((self.n , 1))
        # bias.
        self.b = 0

        for it in range(1, self.it+1):
            # 1. Tìm giá trị dự đoán thông qua hàm giả thuyết
            y_pred = self.hypothesis(self.w, self.b, X)

            # 2. Tìm giá trị hàm theo dõi lỗi
            cost = self.cost_function(y, y_pred)

            #3. Tính đạo hàm ma trận trọng số w
                # X.T: ma trận chuyển vị của X
            dw = (1/self.m) * np.dot(X.T, (y_pred - y)) + self.regularization.derivation(self.w)
            # tính bias mới
            db = -(2 / self.m) * np.sum((y_pred - y))

            # 4. Cập nhật lại tham số trọng lượng và bias
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

            # if it % 1000 == 0:
            #     print("Chi phí cho lần lặp {} là {} :)".format(it, cost))
    def predict(self, test_X):
        # test_X: giá trị cần dự đoán
        # chèn cột gtrị bias=1 vào đầu vector nhiều chiều 
        test_X = np.insert(test_X, 0, 1, axis=1)

        y_pred = self.hypothesis(self.w, self.b, test_X)
        return y_pred
'''Hàm chính quy hóa'''
class l2_regularization:
    #khởi tạo tham số
    def __init__(self, lamda):
        self.lamda = lamda

    def __call__(self, weights):
        return self.lamda * np.sum(np.square(weights))
    
    def derivation(self, weights):
        return self.lamda * 2 * (weights)

class RidgeRegression(Regression):
    def __init__(self, lamda, learning_rate, iteration):
        self.regularization = l2_regularization(lamda)
        super(RidgeRegression, self).__init__(learning_rate, iteration, self.regularization)

    def train(self, X, y):
        return super(RidgeRegression, self).train(X, y)
    def predict(self, test_X):
        return super(RidgeRegression, self).predict(test_X)