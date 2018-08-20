import pandas
import numpy as np
from sklearn.svm import SVC

PATH = '/Users/vlasnikita/Documents/vls/py/ml_py_coursera/'

names = ['y', 'a', 'b']
data = pandas.read_csv(PATH + 'lesson_3/svm-data.csv', header=None, names=names)
y = data['y']
X = data.drop('y', axis=1)

clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(X,y)

# Расстояния до разделяющей гиперплоскости
result = clf.decision_function(X)

# Индексы опорных векторов - объектов, лежащих ближе всего к гиперплоскости
supports = clf.support_
