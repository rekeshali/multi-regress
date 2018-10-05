#!/usr/bin/python

import numpy as np
from scipy import linalg as la, sparse as sp
from prepare import key, keyst, dbst, dbtrain, dbtest

# converts dict into array
def build_X(db, key, goodidx):
    rows = len(db[key[0]])
    cols = len(goodidx) + 1
    X = list(np.ones(rows))
    for idx in goodidx:
        X = X + db[key[idx]]
    X = np.matrix(X)
    X.resize(cols,rows)
    X = X.T
    return X

# extracts output array
def build_R(db, key, goodidx):
    rows = len(db[key[0]])
    R = db[key[goodidx]]
    R = np.matrix(R)
    R.resize(rows,1)
    return R

def linear(dbtrain, key, Xidx, Ridx):
# build arrays from dict using only relevant keys
# Xidx = range(2,5)
#     Xidx = [3,4,5,6]
    Xtrain = build_X(dbtrain, key, Xidx)
#     Xtest  = build_X(dbtest,  key, Xidx)

# build solutions
#     Ridx = 0
    Rtrain = build_R(dbtrain, key, Ridx)
#     Rtest  = build_R(dbtest,  key, Ridx)

# solve linear system
    W = Xtrain.I*Rtrain
    return W, Xtrain
