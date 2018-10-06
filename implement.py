#!/usr/bin/python
import numpy as np

def build_X(db, key, goodidx):
# extracts features of goodidx within key to train model
    rows = len(db[key[0]])
    cols = len(goodidx) + 1
    X = list(np.ones(rows))
    for idx in goodidx:
        X = X + db[key[idx]]
    X = np.matrix(X)
    X.resize(cols,rows)
    X = X.T
    return X

def build_R(db, ukey):
# extracts output array 
    rows = len(db[ukey])
    R = db[ukey]
    R = np.matrix(R)
    R.resize(rows,1)
    return R

# trains a linear regression model
def linear_train(dbtrain, key, Xidx, ukey):
# build arrays from dict using only relevant keys
    Xtrain = build_X(dbtrain, key, Xidx)
# build solutions
    Rtrain = build_R(dbtrain, ukey)
# solve linear system
    W = Xtrain.I*Rtrain
    return W, Xtrain

# solves test array using weights from training
def solve(dbtest, key, Xidx, W):
# build arrays from dict using only relevant keys
    Xtest  = build_X(dbtest,  key, Xidx)
# solve linear system
    R = Xtest*W
    return R, Xtest
