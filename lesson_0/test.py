import numpy as np

# 1
X = np.random.normal(loc=1, scale=10, size=(1000,50))

m = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_norm = ((X - m) / std)

# 2
Z = np.array([[4, 5, 0], 
             [1, 9, 3],              
             [5, 1, 1],
             [3, 3, 3], 
             [9, 9, 9], 
             [4, 7, 1]])

r = np.sum(Z, axis=1)
# print np.nonzero(r > 10)

# 3
a = np.eye(3, 3)
b = np.eye(3, 3)
# print np.vstack((a, b))