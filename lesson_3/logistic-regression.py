import pandas
import numpy as np
from sklearn.metrics import roc_auc_score
import math

PATH = '/Users/vlasnikita/Documents/vls/py/ml_py_coursera/'

data = pandas.read_csv(PATH + 'lesson_3/data-logistic.csv', header=None)
y = data.values[:, :1].T[0]
X = data.values[:, 1:]

def distance (x, y):
    return np.sqrt(np.square(y[0] - x[0]) + np.square(y[1] - x[1]))

def sigmoid (w1, w2, x1, x2):
    return 1 / (1 + np.exp(-w1 * x1 - w2 * x2))

def gradient (w, k, X, y, C, epsilon, max_iterations):
    w_1, w_2 = w
    i = 0
    end_i = 0
    result = []
    
    for i in range(max_iterations):
        end_i = i
        w_1_new = w_1 + k * np.mean(
            y * X[:,0] * (
                1 - 1 / (1 + np.exp(-y * (w_1 * X[:,0] + w_2 * X[:,1])))
            ) - k * C * w_1
        )
        
        w_2_new = w_2 + k * np.mean(
            y * X[:,1] * (
                1 - 1 / (1 + np.exp(-y * (w_1 * X[:,0] + w_2 * X[:,1])))
            ) - k * C * w_2
        )
        
#         print w_1_new, w_2_new

        if distance((w_1, w_2), (w_1_new, w_2_new)) < epsilon:
            break
            
        w_1, w_2 = w_1_new, w_2_new
        
    for j in range(len(y)):
        result.append(sigmoid(w_1, w_2, X[j,0], X[j,1]))

    print 'C: {}, number of iterations: {}'.format(C, end_i)
    return result

w = [0.0, 0.0]
k = 0.1
epsilon = 0.00001
max_iterations = 10000
C = 0
C_l2 = 10

y_score = gradient( w, k, X, y, C, epsilon, max_iterations )
y_score_l2 = gradient( w, k, X, y, C_l2, epsilon, max_iterations )

print round(roc_auc_score(y, y_score), 3)
print round(roc_auc_score(y, y_score_l2), 3)