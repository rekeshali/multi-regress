#!/usr/bin/python
import numpy as np

def flong(features):
    from prepare import keylong
    featureslong = []
    for feat in features:
        for tooth in keylong:
            if feat[0:2] == tooth[0:2]:
                featureslong.append(tooth)
    return featureslong

def buildX(db, features, order, norm):
    # extracts features into matrix for 1st or 2nd order poly regress
    featlong = flong(features)
    rows = len(db[featlong[0]])
    X = list(np.ones(rows)) 
    # first order polynomial
    if order == 1:
        cols = 1 + len(featlong)
        for feat in featlong:
            X = X + db[feat]
    # second order polynomial
    if order == 2:
        if len(featlong) > 1:
            mixlen = np.math.factorial(len(featlong)) / (2*np.math.factorial(len(featlong) - 2))
        else:
            mixlen = 0
        cols = 1 + 2*len(featlong) + mixlen
        for feat in featlong:
            X = X + db[feat] + [a*b for a,b in zip(db[feat], db[feat])]
        i = 0
        j = 1
        for m in range(mixlen):
            X = X + [a*b for a,b in zip(db[featlong[i]],db[featlong[j]])]
            j = j + 1
            if j == len(featlong):
                i = i + 1
                j = i + 1
    # convert types and reshape
    if norm != 'no':
        cols = cols - 1
        X = X[rows:]
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
def train(dbtrain, features, unkey, order, norm):
    Xtrain = buildX(dbtrain, features, order, norm)
    Rtrain = buildR(dbtrain, unkey)
#     W2 = Xtrain.I*Rtrain
    W = np.linalg.pinv(Xtrain.T*Xtrain, rcond=1.0e-15)*Xtrain.T*Rtrain
#     print(W2-W)
#     print(W)
    return W

# solves test array using weights from training
def solve(dbtest, features, order, norm, W):
    # build arrays from dict using only relevant keys
    Xtest  = buildX(dbtest, features, order, norm)
    # solve linear system
    R = Xtest*W
    return R
