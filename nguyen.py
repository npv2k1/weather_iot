import numpy as np
import pandas as pd
import math
import sys

from sklearn.datasets import make_classification
from sklearn.datasets import make_regression

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

"""
1. Lớp DecisionNode: Lớp này dùng để tạo các nút trong cây quyết định. Nó có hai thuộc tính:
a) tạp chất: Thuộc tính này lưu trữ tạp chất của nút. Nó được sử dụng để tính toán mức tăng.
b) câu hỏi: Thuộc tính này lưu trữ câu hỏi được hỏi trong một nút cụ thể. Nó được sử dụng để phân chia tập dữ liệu.
2. Phương thức __init__: Phương thức này khởi tạo đối tượng của lớp DecisionNode. Nó có 6 tham số:
a) tạp chất: Tham số này lưu trữ tạp chất của nút. Nó được sử dụng để tính toán mức tăng.
b) câu hỏi: Tham số này lưu trữ câu hỏi được hỏi trong một nút cụ thể. Nó được sử dụng để phân chia tập dữ liệu.
c) true_subtree: Tham số này lưu trữ cây con bên trái của nút.
d) false_subtree: Tham số này lưu cây con bên phải của nút.
e) Feature_index: Tham số này lưu trữ chỉ mục của tính năng được sử dụng để phân chia tập dữ liệu.
f) ngưỡng: Tham số này lưu trữ giá trị ngưỡng của tính năng được sử dụng để phân tách tập dữ liệu.
  """


class DecisionNode:
    """
    Lớp cho nút cha/lá trong cây quyết định.
    Một nút có thông tin nút về nút trái và nút phải nếu có. nó cũng có thông tin về tạp chất.
    """

    def __init__(self, impurity=None, question=None, feature_index=None, threshold=None,
                 true_subtree=None, false_subtree=None):
        self.impurity = impurity
        # Câu hỏi được hỏi trong nút này để phân chia tập dữ liệu.
        self.question = question
        # Chỉ mục của tính năng phù hợp nhất cho nút này.
        self.feature_index = feature_index
        # Giá trị ngưỡng cho tính năng đó để thực hiện phân tách.
        self.threshold = threshold
        # Trái
        self.true_left_subtree = true_subtree
        # phải
        self.false_right_subtree = false_subtree


class LeafNode:
    """ Nút lá trong cây quyết định. Nó có một giá trị dự đoán."""

    def __init__(self, value):
        self.prediction_value = value


class DecisionTree:
    """
    Cây quyết đinh.
    Cách hoạt động:
    1. Tạo cây rỗng
    2. Tìm ra tính năng phù hợp nhất để phân chia tập dữ liệu
        2.1. Tìm ra ngưỡng phù hợp nhất để phân chia tập dữ liệu
        2.2. Tính toán tạp chất sau khi phân chia (tạp chất là mức tăng thông tin sau khi phân chia)
        2.3. Tìm ra ngưỡng tốt nhất
    3. Tạo cây con bên trái và bên phải
        3.1. Số lượng mẫu nhỏ hơn min_sample_split thì dừng trả về nút lá chứa giá trị dự đoán
    4. Lặp lại bước 2 và 3 cho đến khi đạt được các điều kiện dừng 



    """

    def __init__(self, min_sample_split=3, min_impurity=1e-7, max_depth=float('inf'),
                 impurity_function=None, leaf_node_calculation=None):
        """Hàm khởi tạo
        1. Tạo cây rỗng
        2. Gán các giá trị mặc định cho các tham số
        3. Tạo hàm tính độ nhánh nếu chưa có 
        """
        self.root = None
        self.min_sample_split = min_sample_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.impurity_function = impurity_function
        self.leaf_node_calculation = leaf_node_calculation

    def _partition_dataset(self, Xy, feature_index, threshold):
        """Tách tập dữ liệu dựa trên tính năng và ngưỡng đã cho.

        """
        split_func = None
        if isinstance(threshold, int) or isinstance(threshold, float):
            # Đối với các ngưỡng số nguyên hoặc thực, hàm phân chia tập dữ liệu
            # sẽ trả về giá trị đúng nếu giá trị của tính năng lớn hơn hoặc bằng
            # ngưỡng đã cho.
            def split_func(sample): return sample[feature_index] >= threshold
        else:
            # Đối với các ngưỡng không phải số nguyên hoặc thực, hàm phân chia tập dữ liệu
            # sẽ trả về giá trị đúng nếu giá trị của tính năng bằng ngưỡng đã cho.
            def split_func(sample): return sample[feature_index] == threshold

        # Tách tập dữ liệu dựa trên hàm phân chia được định nghĩa.
        X_1 = np.array([sample for sample in Xy if split_func(sample)])
        X_2 = np.array([sample for sample in Xy if not split_func(sample)])

        return X_1, X_2

    def _find_best_split(self, Xy):
        """ Tìm ngưỡng tốt nhất giúp phân chia dữ liệu tốt.

        """
        # cái này sẽ chứa tính năng và giá trị của nó để tạo ra sự phân chia tốt nhất (higest gain).
        best_question = tuple()

        # best data split.
        best_datasplit = {}

        largest_impurity = 0
        n_features = (Xy.shape[1] - 1)
        # Lặp tất cả các đăc trưng để tìm ra ngưỡng tốt nhất.
        for feature_index in range(n_features):
            # lấy tất cả các giá trị duy nhất của đặc trưng.
            unique_value = set(s for s in Xy[:, feature_index])

            # lặp qua tất cả các giá trị duy nhất của đặc trưng.
            for threshold in unique_value:
                # chia tập dữ liệu dựa trên giá trị tính năng.
                true_xy, false_xy = self._partition_dataset(
                    Xy, feature_index, threshold)
                # bỏ qua nút có bất kỳ loại 0. vì điều này có nghĩa là nó đã thuần.
                if len(true_xy) > 0 and len(false_xy) > 0:

                    # find the y values.
                    y = Xy[:, -1]
                    true_y = true_xy[:, -1]
                    false_y = false_xy[:, -1]

                    # calculate the impurity function.
                    impurity = self.impurity_function(y, true_y, false_y)

                    # if the calculated impurity is larger than save this value for comaparition.
                    if impurity > largest_impurity:
                        largest_impurity = impurity
                        best_question = (feature_index, threshold)
                        best_datasplit = {
                            # X of left subtree
                            "leftX": true_xy[:, :n_features],
                            # y of left subtree
                            "lefty": true_xy[:, n_features:],
                            # X of right subtree
                            "rightX": false_xy[:, :n_features],
                            # y of right subtree
                            "righty": false_xy[:, n_features:]
                        }

        return largest_impurity, best_question, best_datasplit

    def _build_tree(self, X, y, current_depth=0):
        """
        Sử dụng đệ quy để xây dựng cây quyết định
        """
        n_samples, n_features = X.shape  # Lấy số lượng mẫu và số lượng đặc trưng

        # Thêm cột y vào X
        Xy = np.concatenate((X, y), axis=1)

        # tìm mức tăng Thông tin trên từng tính năng, từng giá trị và trả về câu hỏi phân tách dữ liệu rất tốt
        # based on the impurity function. (classfication (Phân lớp) - Information gain, regression (Hồi quy) - variance reduction).
        if (n_samples >= self.min_sample_split) and (current_depth <= self.max_depth):
            # tìm cách phân chia tốt nhất/câu hỏi nào phân chia dữ liệu tốt.
            impurity, quesion, best_datasplit = self._find_best_split(Xy)
            if impurity > self.min_impurity:
                # Xây dựng các cây con cho nhánh phải và nhánh trái.
                true_branch = self._build_tree(
                    best_datasplit["leftX"], best_datasplit["lefty"], current_depth + 1)
                false_branch = self._build_tree(
                    best_datasplit["rightX"], best_datasplit["righty"], current_depth + 1)
                return DecisionNode(impurity=impurity, question=quesion, feature_index=quesion[0], threshold=quesion[1],
                                    true_subtree=true_branch, false_subtree=false_branch)

        leaf_value = self._leaf_value_calculation(y)
        return LeafNode(value=leaf_value)

    def train(self, X, y):
        """
        Xây dựng cây quyết đinh

        :param X: train features values .
        :param y: train targetvalue.
        """
        self.root = self._build_tree(X, y, current_depth=0)

    def predict_sample(self, x, tree=None):
        """move form the top to bottom of the tree make a prediction of the sample by the
            value in the leaf node """
        if tree is None:
            tree = self.root
        # if it a leaf node the return the prediction.
        if isinstance(tree, LeafNode):

            return tree.prediction_value
        feature_value = x[tree.feature_index]

        branch = tree.false_right_subtree

        if isinstance(feature_value, int) or isinstance(feature_value, float):

            if feature_value >= tree.threshold:

                branch = tree.true_left_subtree
        elif feature_value == tree.threshold:
            branch = tree.true_left_subtree

        return self.predict_sample(x, branch)

    def predict(self, test_X):
        """ predict the unknow feature."""
        x = np.array(test_X)
        y_pred = [self.predict_sample(sample) for sample in x]
        # y_pred = np.array(y_pred)
        # y_pred = np.expand_dims(y_pred, axis = 1)
        return y_pred

    def draw_tree(self, tree=None, indentation=" "):
        """print the whole decitions of the tree from top to bottom."""
        if tree is None:
            tree = self.root

        def print_question(question, indention):
            """
            :param question: tuple of feature_index and threshold.
            """
            feature_index = question[0]
            threshold = question[1]

            condition = "=="
            if isinstance(threshold, int) or isinstance(threshold, float):
                condition = ">="
            print(indention, "Is {col}{condition}{value}?".format(
                col=feature_index, condition=condition, value=threshold))

        if isinstance(tree, LeafNode):
            print(indentation, "The predicted value -->", tree.prediction_value)
            return

        else:
            # print the question.
            print_question(tree.question, indentation)
            if tree.true_left_subtree is not None:
                # travers to the true left branch.
                print(indentation + '----- True branch :)')
                self.draw_tree(tree.true_left_subtree, indentation + "  ")
            if tree.false_right_subtree is not None:
                # travers to the false right-side branch.
                print(indentation + '----- False branch :)')
                self.draw_tree(tree.false_right_subtree, indentation + "  ")


class DecisionTreeRegression(DecisionTree):
    """ Decision Tree for the Regression problem."""

    def __init__(self, min_sample_split=3, min_impurity=1e-7, max_depth=float('inf'),
                 ):
        """
        :param min_sample_split: min value a leaf node must have.
        :param min_impurity: minimum impurity.
        :param max_depth: maximum depth of the tree.
        """
        self._impurity_function = self._claculate_variance_reduction
        self._leaf_value_calculation = self._calculate_colum_mean
        super(DecisionTreeRegression, self).__init__(min_sample_split=min_sample_split, min_impurity=min_impurity, max_depth=max_depth,
                                                     impurity_function=self._impurity_function, leaf_node_calculation=self._leaf_value_calculation)

    def _claculate_variance_reduction(self, y, y1, y2):
        """
        Tính Giảm phương sai Variance Reduction.

        :param y: target value.
        :param y1: target value for dataset in the true split/right branch.
        :param y2: target value for dataset in the false split/left branch.
        """
        # propobility of true values.
        variance = np.var(y)
        variance_y1 = np.var(y1)
        variance_y2 = np.var(y2)

        y_len = len(y)
        fraction_1 = len(y1) / y_len
        fraction_2 = len(y2) / y_len
        variance_reduction = variance - \
            (fraction_1 * variance_y1 + fraction_2 * variance_y2)
        return variance_reduction

    def _calculate_colum_mean(self, y):
        """
        Tính toán giá trị nút là giá trị trung bình của các giá trị.

        :param y: leaf node target array.
        """
        mean = np.mean(y, axis=0)
        return mean

    def train(self, X, y):
        """
        Build the tree.

        :param X: Feature array/depentant values.
        :parma y: target array/indepentant values.
        """
        # train the model.
        super(DecisionTreeRegression, self).train(X, y)
