#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as mp

from explore import car, key
from prepare import prepare, randsplit, stat
# from implement import build_X, build_R, linear_train, solve # for debugging
from tools import auto, error, topcombo, counthits
from output import print_results
# from reduct import scatter 

[db, dbst, keyst] = prepare()
xinp = range(1,8)
unk  = 0
topN = 20
Iters = 10000
for j in range(10):
    errmean  = 0
    keycount = np.zeros(len(key))
    for i in range(Iters):
        [dbtrain, dbtest] = randsplit(db)
        [F, R] = auto(dbtrain, dbtest, xinp, unk)
        dberr  = error(dbtest, F, R)
        dbtop  = topcombo(dberr, topN)
        errmean  = errmean + np.mean(dbtop['err'])
        keycount = keycount + counthits(dbtop)
    errmean  = errmean/Iters
    keycount = keycount/(topN*Iters)

    print_results(Iters, errmean, keycount)
