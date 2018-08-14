import pandas
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

PATH = '/Users/vlasnikita/Documents/vls/py/ml_py_coursera/'

# Генератор разбиений для обучения: train=k-1, target=1; перемешивает выборку
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Имена колонок, где 'y' - название колонки с классами, все остальные - название признаков
names = ['y', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315', 'Proline']
data = pandas.read_csv(PATH + 'lesson_2/wine.data', header=None, names=names)

y = data['y']

# I. Без масштабирования
X = data.drop('y', axis=1)
result = []

for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X, y)
    result.append({
        'number': i,
        'value': np.around(np.mean(cross_val_score(knn, X, y, cv=kf)), decimals=2)
    })

sorted_result = sorted(result, key=lambda x: x['value'], reverse=True)

# 1
task_1 = open(PATH + 'lesson_2/1.txt', 'w')
task_1.write(str(sorted_result[0]['number']))
# 2
task_2 = open(PATH + 'lesson_2/2.txt', 'w')
task_2.write(str(sorted_result[0]['value']))

print sorted_result

# II. С масштабированием
standardized_X = preprocessing.scale(X)
standardized_result = []

for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(standardized_X, y)
    standardized_result.append({
        'number': i,
        'value': np.around(np.mean(cross_val_score(knn, standardized_X, y, cv=kf)), decimals=2)
    })

standardized_sorted_result = sorted(standardized_result, key=lambda x: x['value'], reverse=True)

# 3
task_1 = open(PATH + 'lesson_2/3.txt', 'w')
task_1.write(str(standardized_sorted_result[0]['number']))
# 4
task_2 = open(PATH + 'lesson_2/4.txt', 'w')
task_2.write(str(standardized_sorted_result[0]['value']))

print standardized_sorted_result