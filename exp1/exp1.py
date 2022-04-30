# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 23:26:19 2022

@author: SinTA
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from cvxpy.atoms import norm
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

data1 = pd.read_csv('cross_before_202108.csv').replace(np.inf, 0)
data2 = pd.read_csv('cross_before_202109.csv').replace(np.inf, 0)

TRAIN = False #True

# unsupervised low-level
X = data2.values[:,:-1].T # column vectors for unlabled features
n, k = X.shape
s = 8
beta = cp.Parameter(nonneg=True, value=1)
# B n,s
# A s.k

def sub_p1(l, i, B_fixed):
    A = cp.Variable((s,i))
    objective = cp.Minimize(
        cp.sum_squares(X[:,l:(l+i)] - B_fixed @ A)
        )
    prob = cp.Problem(objective, [])
    result = prob.solve()
    return result, A.value

def sub_p2(l, i, A_fixed):
    B = cp.Variable((n,s))
    objective = cp.Minimize(
        cp.sum_squares(X[:,l:(l+i)] - B @ A_fixed)
        )
    constraints = [norm(B,2,1) <= 1]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(    )
    return result, B.value

i, lr = 100, 1/1000
np.random.seed(1234)
B_fixed = np.random.normal(1,s,size = (n,s))
A_fixed = np.random.normal(1,s,size = (s,i))
inner_max = 5
outer_max = 10

if TRAIN:
    for _ in range(outer_max):
        result1s = []
        result2s = []
        for l in tqdm(range(0, k-i,i//2)):
            for _ in range(inner_max):
                try:
                    result1, A_fixed_new = sub_p1(l, i, B_fixed)
                    result2, B_fixed_new = sub_p2(l, i, A_fixed)
                except:
                    lr /=1.01
                    continue
                if (type(A_fixed_new) == type(None)) or (type(B_fixed_new) == type(None)):
                    break
                result1s.append(result1)
                result2s.append(result2)
                B_fixed = (1-lr) * B_fixed + lr * B_fixed_new
                A_fixed = (1-lr) * A_fixed + lr * A_fixed_new
                A_fixed_new, B_fixed_new = None, None
            A_fixed[:,0:(i//2)] = A_fixed[:,(i//2):]
        print('\n',np.mean(result1), np.mean(result2))
    np.save('B_fixed', B_fixed)
else:
    B_fixed = np.load('B_fixed.npy')
    
# step 2, for labeld data
def getA(X, B_fixed):
    A = cp.Variable((s,X.shape[1]))
    objective = cp.Minimize(
        cp.sum_squares(X - B_fixed @ A)
        )
    prob = cp.Problem(objective, [])
    result = prob.solve()
    return A.value

X_l = data1.values[:,:-1].T # column vectors for labled features

A_l = getA(X_l, B_fixed).T
A_u = getA(X, B_fixed).T
y_l = data1.values[:,-1] 
y_u = data2.values[:,-1] 

clf = LinearRegression().fit(A_l, y_l)


clf2 = LinearRegression().fit(X_l.T, y_l)

# simply select profitable stocks and calc total return 
print('factor model return(%)', np.mean([y_u * 100 for x, y_u in zip(clf.predict(A_u),y_u) if x>0]) )
print('transfer learning return(%)', np.mean([y_u * 100 for x, y_u in zip(clf2.predict(A_u),y_u) if x>0]))
