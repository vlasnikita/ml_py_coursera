import pandas
import numpy as np
from itertools import imap
import re
from collections import Counter

PATH = '/Users/vlasnikita/Documents/vls/py/ml_py_coursera/'

data = pandas.read_csv(PATH + 'titanic.csv', index_col='PassengerId')
total = float(data.shape[0])
# 1
# # 1.1
print 'MALES ', data[data.Sex == u'male'].shape[0]
# # 1.2
print 'FEMALES ', data[data.Sex == u'female'].shape[0]

# 2
survived = float(data[data.Survived == 1].shape[0])
print 'SURVIVED, % ', round(survived / total * 100, 2)

# 3
firstClass = float(data[data.Pclass == 1].shape[0])
print 'FIRST CLASS, % ', round(firstClass / total * 100, 2)

# 4
# # 4.1
haveAge = np.array(data[data.Age > 0])[:,4]
print 'MEAN AGE ', round(np.mean(haveAge, axis=0, dtype=np.float64), 2)
print 'MEDIUM AGE', (haveAge[len(haveAge) / 2] + haveAge[len(haveAge) / 2 - 1]) / 2 

# 5
def pearsonr(x,y):
    n = len(x)
    
    sum_x = float(sum(x))
    sum_y = float(sum(y))
    
    sum_x_sq = sum(map(lambda x: pow(x, 2), x))
    sum_y_sq = sum(map(lambda x: pow(x, 2), y))

    psum = sum(imap(lambda x,y: x * y, x, y))
    num = psum - (sum_x * sum_y / n)
    den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
    if den == 0: return 0
    return num / den

sibsp = np.array(data)[:,5].astype(float)
parch = np.array(data)[:,6].astype(float)

customPearson = round(pearsonr(sibsp, parch), 2)
numpyPearson = round(np.corrcoef(sibsp, parch)[0,1], 2)

print 'CUSTOM PEARSON: ', customPearson
print 'NUMPY PEARSON: ', numpyPearson

# 6
def name_formatter(str):
    sub_s = 'Mrs.'
    sub_fn = '('

    if (sub_s in str) and (sub_fn in str):
        return re.search(r'\(+(\S+)', str).group(1)
    else:
        return re.search(r'\.\s(\S+)', str).group(1)

women = np.array(data[data.Sex == u'female'])[:,2]
women_formatted = map(name_formatter, women)
print Counter(women_formatted)