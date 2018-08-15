import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor

PATH = '/Users/vlasnikita/Documents/vls/py/ml_py_coursera/'

boston = load_boston()
X = scale(boston['data'])
y = boston['target']
kf = KFold(n_splits=5, shuffle=True, random_state=42)
result = []

for i, val in np.ndenumerate(np.linspace(1, 10, num=200)):
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance', p=val)
    knn.fit(X, y)
    result.append({
        'p': val,
        'score': np.mean(cross_val_score(knn, X, y, cv=kf, scoring='neg_mean_squared_error'))
    })

best = sorted(result, key=lambda x: x['score'], reverse=True)

file = open(PATH + 'lesson_2/II_1.txt', 'w')
file.write(str(result[0]['p']))