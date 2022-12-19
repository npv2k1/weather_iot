import numpy as np
import pandas as pd

from sklearn.datasets import make_regression

from sklearn.metrics import r2_score

class LinearRegression:
    def __init__(self, learning_rate, iteration):
        """
        :param learning_rate: phần trăm học thuật cho một lần lặp, 0,1.
        :param iteration: số lần lặp đào tạo.
        self.m : so mau du lieu
        self.n: so thuoc tinh
        self.w: trong so
        """
        self.m = None
        self.n = None
        self.w = None
        self.lr = learning_rate
        self.it = iteration
    
    # đánh giá lỗi của thuật toán --> để xem giá trị của hàm số có giảm theo mỗi vòng lặp hoặc không
    def cost_function(self, y, y_pred):
        """
        :param y: Gia tri muc tieu ban dau.
        :param y_pred: Gia tri muc tieu du doan.
        """
        return (1 / (2*self.m)) * np.sum(np.square(y_pred - y)) # hàm lỗi: (1/(2*m)) * sum((y_pred-y)^2)
    
    def hypothesis(self, w, X):
        """
        :param w: trọng số giá trị tham số.
        :param X: Mẫu đào tạo.
        """
        return np.dot(X, w) # tích vô hướng của hai vector X và w --> 1 number

    def train(self, X, y):
        """
        :param X: training data feature values ---> vector n chieu.
        :param y: training data target value -----> mang 1 chieu.
        """
        # chèn thêm trọng số có giá trị 1 vào vị trí 0 cho X
        X = np.insert(X, 0, 1, axis=1)
        # Giá trị mục tiêu phải ở dạng (n, 1) chứ không phải (n, ).
        # Vì vậy, điều này sẽ kiểm tra điều đó và thay đổi hình dạng thành (n, 1), nếu không có.
        try:
            y.shape[1]
        except IndexError as e:
            # cần thay đổi nó thành mảng 1 D, không phải danh sách.
            print("ERROR: Target array should be a one dimentional array not a list"
                  "----> here the target value not in the shape of (n,1). \nShape ({shape_y_0},1) and {shape_y} not match"
                  .format(shape_y_0 = y.shape[0] , shape_y = y.shape))
            return 
        
        # m là số lượng mẫu để trainning
        self.m = X.shape[0]
        # n là số lượng các features.
        self.n = X.shape[1]

        # Đặt trọng số ban đầu là một mảng 2 chiều với n = 8 ,m = 1 và các phần tử trong mảng = 0.
        self.w = np.zeros((self.n , 1)) 

        for it in range(1, self.it+1):
            # 1. Tìm giá trị dự đoán thông qua giả thuyết.
            # 2. Tìm giá trị hàm Cost.
            # 3. Tìm đạo hàm của trọng số.
            # 4. Áp dụng Gradient Decent.
            y_pred = self.hypothesis(self.w, X) 

            cost = self.cost_function(y, y_pred)
            # ma trận đạo hàm
            dw = (1/self.m) * np.dot(X.T, (y_pred - y)) 

            # thay đổi trọng số
            self.w = self.w - self.lr * dw

            # if it % 1000 == 0:
            #     print("The Cost function for the iteration {}----->{} :)".format(it, cost))
    def predict(self, test_X):
        """
        :param test_X: giá trị đặc trưng để dự đoán.
        """
        # chèn thêm trọng số có giá trị 1 vào vị trí 0 cho test_X
        test_X = np.insert(test_X, 0, 1, axis=1)
        
        # tìm giá trị dự đoán thông qua hàm train đã đc gọi --> W 
        y_pred = self.hypothesis(self.w, test_X)
        return y_pred