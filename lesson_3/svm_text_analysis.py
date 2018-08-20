import pandas
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )

# Переводит текст в вектора
vectorizer = TfidfVectorizer() 


X_train = vectorizer.fit_transform(newsgroups.data)
X_test = vectorizer.transform(newsgroups.data)
y = newsgroups.target

# Сетка значений параметров
grid = {'C': np.power(10.0, np.arange(-5, 6))}

# Кросс-валидатор на k-1 / 1 выборке
kf = KFold(n_splits=5, shuffle=True, random_state=241)

# Линейный SVM-классификатор 
clf = SVC(kernel='linear', random_state=241)

# Подбор параметров по сетке
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=kf)

gs.fit(X_train, y)

# Лучший минимальный параметр С. Спойлер: дефолтная единица (1.0)
C_best = gs.best_params_.get('C')

results = gs.best_estimator_.coef_ 

row = results.getrow(0).toarray()[0].ravel()
top_ten_indicies = np.argsort(abs(row))[-10:]
top_ten_values = row[top_ten_indicies]

feature_mapping = vectorizer.get_feature_names()

top_ten_words = []

for a in top_ten_indicies:
    top_ten_words.append(feature_mapping[a])

print sorted(top_ten_words)
# clf_with_best_minimal_C = SVC(kernel='linear', random_state=241, C=1)
# clf_with_best_minimal_C.fit(X_train, y)

# weights = clf_with_best_minimal_C.coef_.toarray()[0]
# result = []

# for i in range(len(weights)):
#     result.append({
#         'index': i,
#         'value': np.absolute(weights[i])
#     })

# sorted_result = sorted(result, key=lambda x: x['value'], reverse=True)

# for a in sorted_result[:10]:
#     print feature_mapping[a['index']]
