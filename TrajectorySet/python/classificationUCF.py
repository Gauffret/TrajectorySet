# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:24:22 2016

@author: matsui
"""
import pickle
import numpy as np
import random as r
from sklearn import preprocessing
"""
with open('../result/101/fisher_dict.pickle', 'r') as f:
    A = pickle.load(f)#train test reverse

max_abs_scaler = preprocessing.MaxAbsScaler()

lis_name = []
lis_value =[]

for k, v in sorted(A.items()):
    lis_name.append(k)
    v_mas = max_abs_scaler.fit_transform(v)
    lis_value.append(v_mas) 


y = np.array([0]*lis_value[0].shape[0])
X = lis_value[0]

for i in range(len(lis_value)-1):
    y = np.hstack((y, np.array([i+1]*lis_value[i+1].shape[0])))
    X = np.vstack((X, lis_value[i+1]))


np.save("../result/101/X_class.npy", X)
np.save("../result/101/y_class.npy", y)
"""

X = np.load("../result/101/X_class.npy")
y = np.load("../result/101/y_class.npy")



z = np.loadtxt("../result/101/fisher_group.txt")
from sklearn.svm import SVC, LinearSVC


from sklearn.neural_network import MLPClassifier

g = open("../result/101/sortSplit2.txt",'w')

from sklearn.model_selection import LeaveOneGroupOut
lol = LeaveOneGroupOut()

scores = []

"""
#Classification random (for UCF50)
for train_index, test_index in lol.split(X, y, z):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = MLPClassifier(verbose = True)
    clf.fit(X_train, y_train)
    
    score = clf.score(X_test, y_test)
    print(score)
    g.write(str(score))
    g.write("\n")
    scores.append(score)

scores = np.array(scores)
print(scores.mean())
g.write("\n"+"mean"+"\n" +str(scores.mean()))

g.close()

"""

#Classification split (for UCF101)
for i in range(3):
    

    train_index = []
    test_index = []
    index = 0
    for grp in z:
        if (i ==0):
            if (grp > 7):
                train_index.append(index)
            else :
                test_index.append(index) 
        if (i ==1):
            if ((grp > 14) or (grp < 8)):
                train_index.append(index)
            else :
                test_index.append(index) 
        if (i ==2):
            if ((grp > 21) or (grp < 15)):
                train_index.append(index)
            else :
                test_index.append(index) 
        index = index + 1

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = MLPClassifier(verbose = True)
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    print(score)
    g.write(str(score))
    g.write("\n")
    scores.append(score)

scores = np.array(scores)
print(scores.mean())
g.write("\n"+"mean"+"\n" +str(scores.mean()))

g.close()