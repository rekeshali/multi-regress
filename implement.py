#!/usr/bin/python
import numpy as np

# def build_X(db, key, goodidx):
#     # extracts features of goodidx within key to train model
#     rows = len(db[key[0]])
#     cols = len(goodidx) + 1
#     X = list(np.ones(rows))
#     for idx in goodidx:
#         X = X + db[key[idx]]
#     X = np.matrix(X)
#     X.resize(cols,rows)
#     X = X.T
#     return X

# # trains a linear regression model
# def linear_train(dbtrain, key, Xidx, ukey):
#     # build arrays from dict using only relevant keys
#     Xtrain = build_X(dbtrain, key, Xidx)
#     # build solutions
#     Rtrain = build_R(dbtrain, ukey)
#     # solve linear system
#     W = Xtrain.I*Rtrain
#     #     W = (Xtrain.T*Xtrain).I*Xtrain.T*Rtrain
#     return W, Xtrain


def buildX(db, features, order):
    # extracts features into matrix for 1st or 2nd order poly regress
    rows = len(db[features[0]])
    X = list(np.ones(rows))
    # first order polynomial
    if order == 1:
        cols = 1 + len(features)
        for feat in features:
            X = X + db[feat]
    # second order polynomial
    if order == 2:
        mixlen = np.math.factorial(len(features)) / (2*np.math.factorial(len(features) - 2))
        cols = 1 + 2*len(features) + mixlen
        for feat in features:
            X = X + db[feat] + [a*b for a,b in zip(db[feat], db[feat])]
        i = 0
        j = 1
        for m in range(mixlen):
            X = X + [a*b for a,b in zip(db[features[i]],db[features[j]])]
            j = j + 1
            if j == len(features):
                i = i + 1
                j = i + 1
    X = np.matrix(X)
    X.resize(cols,rows)
    X = X.T
    return X

def buildR(db, ukey):
# extracts output array 
    rows = len(db[ukey])
    R = db[ukey]
    R = np.matrix(R)
    R.resize(rows,1)
    return R

# trains an n polynomial model with any feature combination 
def train(dbtrain, features, unkey, order):
    Xtrain = buildX(dbtrain, features, order)
    Rtrain = buildR(dbtrain, unkey)
    W = Xtrain.I*Rtrain
    return W

# solves test array using weights from training
def solve(dbtest, features, order, W):
    # build arrays from dict using only relevant keys
    Xtest  = buildX(dbtest, features, order)
    # solve linear system
    R = Xtest*W
    return R
