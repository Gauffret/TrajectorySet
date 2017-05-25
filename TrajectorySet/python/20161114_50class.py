# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:24:22 2016

@author: matsui
"""
import pickle
import numpy as np

with open('/home/matsui/only_tra/50fisher/fisher_dict1.pickle', 'r') as f:
    A = pickle.load(f)#train test reverse
with open('/home/matsui/only_tra/50fisher/fisher_dict2.pickle', 'r') as f:
    B = pickle.load(f)#train test reverse
with open('/home/matsui/only_tra/50fisher/fisher_dict3.pickle', 'r') as f:
    C = pickle.load(f)#train test reverse
lis_name = []
lis_value =[]
for k, v in A.items():
    lis_name.append([k[42:]])
    lis_value.append(v) 

for k, v in B.items():
    lis_name.append([k[43:]])
    lis_value.append(v) 

for k, v in C.items():
    lis_name.append([k[43:]])
    lis_value.append(v) 

    
y = np.array([0]*lis_value[0].shape[0])
X = lis_value[0]

for i in range(len(lis_value)-1):

    y = np.hstack((y, np.array([i+1]*lis_value[i+1].shape[0])))
    X = np.vstack((X, lis_value[i+1]))


np.save("../50fisher/X_50class.npy", X)
np.save("../50fisher/y_50class.npy", y)

z1 = np.loadtxt("../50fisher/fisher_group1.txt")
z2 = np.loadtxt("../50fisher/fisher_group2.txt")
z3 = np.loadtxt("../50fisher/fisher_group3.txt")

z = np.hstack((z1,z2,z3))

from sklearn.svm import SVC, LinearSVC

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(verbose=True)


from sklearn.model_selection import LeaveOneGroupOut
lol = LeaveOneGroupOut()

scores = []
for train_index, test_index in lol.split(X, y, z):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf = MLPClassifier(verbose=True)
    clf.fit(X_train, y_train)
    
    score = clf.score(X_test, y_test)
    print(score)
    scores.append(score)

scores = np.array(scores)
print(scores.mean())
