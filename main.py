#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as mp

from explore import car, key
from prepare import prepare, randsplit, stat
from tools import auto, error, topcombo, counthits, initdball, bestcombo, findball
from output import print_results, print_max

[db, dbst, keyst] = prepare(1)
xinp  = range(1,8)
unk   = 0
order = 2
topN  = 5
Iters = 100
tests = 5
lol = 0
for j in range(tests):
    errmean  = 0
    keycount = np.zeros(len(key))
    dball = initdball()
    for i in range(Iters):
        [dbtrain, dbtest] = randsplit(db)
        [F, R] = auto(dbtrain, dbtest, xinp, unk, order)
        dberr  = error(dbtest, F, R)
        dbtop  = topcombo(dberr, topN)
        errmean  = errmean + np.mean(dbtop['err'])
        keycount = keycount + counthits(dbtop)
        [dball, lol] = bestcombo(dbtop['fts'][0], dbtop['err'][0], dball, lol)
    errmean  = errmean/Iters
    keycount = keycount/(topN*Iters)
    dbmax = findball(dball)
    print_results(order, Iters, errmean, keycount)
    print_max(order, Iters, dbmax)
