from sklearn import metrics
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_predict, cross_val_score, KFold, \
    StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import metrics


def train_test(data_train_feature, data_target):
    kfold = StratifiedKFold(n_splits=9, shuffle=True, random_state=0)
    train_test_index = []
    for train_index, test_index in kfold.split(data_train_feature, data_target):
        train_test_index.append([train_index, test_index])
    return train_test_index


class Data:
    _X_train = None
    _Y_train = None

    def __init__(self, data_feature, data_target, train_index):
        # Extract the no of features
        self.noOfFeatures = data_feature.shape[1] - 1
        X_train = data_feature[train_index, 0:-1]
        minmax = MinMaxScaler()
        # Extract all the features to _X
        self._X_train = minmax.fit_transform(X_train)
        self._Y_train = data_target[train_index]
        self.train_test = train_test(data_feature[train_index], data_target[train_index])

    def getTrainAccuracy(self, features):
        acc = []
        for train_index, test_index in self.train_test:
            dTree = DecisionTreeClassifier()
            train = self._X_train[train_index, 0:-1]
            test = self._X_train[test_index, 0:-1]
            dTree.fit(train[:, features], self._Y_train[train_index])
            y_pred = dTree.predict(test[:, features])
            y_true = self._Y_train[test_index]
            acc.append(metrics.balanced_accuracy_score(y_pred=y_pred, y_true=y_true))
        return np.mean(acc)

    def getDimension(self):
        return self.noOfFeatures


class Test_Data:
    _X_test = None
    _Y_test = None

    def __init__(self, data_train_feature, data_target, train_index, test_index):
        # Extract the no of features
        self.noOfFeatures = data_train_feature.shape[1] - 1
        minmax = MinMaxScaler()

        train = data_train_feature[train_index, 0: - 1]
        test = data_train_feature[test_index, 0: - 1]

        minmax.fit(train)
        # Extract all the features to _X
        self._X_train = minmax.transform(train)
        self._X_test = minmax.transform(test)
        self._Y_train = data_target[train_index]
        self._Y_test = data_target[test_index]

    def getTestAccuracy(self, features):
        knn = DecisionTreeClassifier()
        knn.fit(self._X_train[:, features], self._Y_train)
        # 测试集预测得到的结果存储在列表中统一计算准确度
        y_pred = knn.predict(self._X_test[:, features])
        return metrics.balanced_accuracy_score(y_pred=y_pred, y_true=self._Y_test)

    def getTestF1(self, features):
        knn = DecisionTreeClassifier()
        knn.fit(self._X_train[:, features], self._Y_train)
        y_pred = knn.predict(self._X_test[:, features])
        return metrics.f1_score(y_pred=y_pred, y_true=self._Y_test, average='micro')

    def getDimension(self):
        return self.noOfFeatures
