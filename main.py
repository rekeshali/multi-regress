#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as mp

from explore import car, key
from prepare import keylong, prepare, stats, znorm, randsplitnorm, seestats
from tools import auto, error, topcombo, counthits, initdball, bestcombo, findball
from output import print_results, print_max

norms = ['no', 'tog', 'dtog', 'sep', 'dsep']
norm  = norms[1]
db = prepare()
xinp  = range(1,8)
unk   = 0
order = 1
topN  = 10
Iters = 10
tests = 1
lol = 0
for j in range(tests):
    errmean  = 0
    keycount = np.zeros(len(key))
    dball = initdball()
    for i in range(Iters):
        [dbtrain, dbtest] = randsplitnorm(db, norm)
        [F, R] = auto(dbtrain, dbtest, xinp, unk, order)
        dberr  = error(dbtest, F, R)
        dbtop  = topcombo(dberr, topN)
        errmean  = errmean + np.mean(dbtop['err'])
        keycount = keycount + counthits(dbtop)
        [dball, lol] = bestcombo(dbtop['fts'][0], dbtop['err'][0], dball, lol)
    errmean  = errmean/Iters
    keycount = keycount/(topN*Iters)
    dbmax = findball(dball)
    print_results(order, Iters, norm, errmean, keycount)
    print_max(order, Iters, norm, dbmax)
