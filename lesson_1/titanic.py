import pandas
import numpy as np
from sklearn import tree
import graphviz

PATH = '/Users/vlasnikita/Documents/vls/py/ml_py_coursera/'

data = pandas.read_csv(PATH + 'titanic.csv', index_col='PassengerId')
data = data[['Survived', 'Pclass', 'Fare', 'Age', 'Sex']]
data = data.dropna(axis=0)

X = data[['Pclass', 'Fare', 'Age', 'Sex']].copy()
sex_mapping = {'male': 0, 'female': 1}
X = X.replace({'Sex': sex_mapping})

y = data[['Survived']]

clf = tree.DecisionTreeClassifier(random_state=241)
clf.fit(X,y)

importances = np.array(clf.feature_importances_)

print 'Pclass: ', importances[0], '\nFare: ', importances[1], '\nAge: ', importances[2], '\nSex: ', importances[3]

dot_data = tree.export_graphviz(
    clf, 
    out_file=None,
    feature_names=['Pclass', 'Fare', 'Age', 'Sex'],
    filled=True, 
    rounded=True,  
    special_characters=True,
    class_names=True
)

graph = graphviz.Source(dot_data)
graph.render("./ml_py_coursera/lesson_1/tree")
