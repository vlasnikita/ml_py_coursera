import pandas
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

PATH = '/Users/vlasnikita/Documents/vls/py/ml_py_coursera/'

names = ['y', 'X_a', 'X_b']

def getAccuracy(X_train, y_train, X_test, y_test):
    clf = Perceptron(random_state=241)
    clf.fit(X_train, y_train)
    return accuracy_score(y_test, clf.predict(X_test))

def scale(X_train, X_test):
    X_scaler = StandardScaler()
    return X_scaler.fit_transform(X_train), X_scaler.transform(X_test)

# 1.1
train_data = pandas.read_csv(PATH + 'lesson_2/perceptron-train.csv', header=None, names=names)
y_train = train_data['y']
X_train = train_data.drop('y', axis=1)

test_data = pandas.read_csv(PATH + 'lesson_2/perceptron-test.csv', header=None, names=names)
y_test = test_data['y']
X_test = test_data.drop('y', axis=1)

result = getAccuracy(X_train, y_train, X_test, y_test)

# 1.2
X_train_scaled, X_test_scaled = scale(X_train, X_test)

result_scaled = getAccuracy(X_train_scaled, y_train, X_test_scaled, y_test)

# 1.3
print round(result_scaled - result, 3)
# print 'Normal: {}\nScaled: {}'.format(test_data, test_data_scaled)
# print 'y: {} \nX: {}\ny_test: {}'.format(y, X, y_test)