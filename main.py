#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as mp

from explore import key, explore, key
from prepare import keylong, prepare, stats, randsplitnorm, seestats
from tools import auto, error, topcombo, counthits, initdball, bestcombo, findball
from output import print_results, print_max

norms = ['no', 'tog', 'dtog', 'sep', 'dsep']
norm  = norms[1]
[dbraw, bad_entry] = explore()
db = prepare(dbraw, bad_entry)
dbstats = stats(db, keylong)
mpgmean = list(dbstats['mpg']['mean'])[0]
mpgstd = list(dbstats['mpg']['std'])[0]

# print(db)
xinp  = range(1,8)
unk   = 0
order = 2
topN  = 5
Iters = 100
tests = 2

for j in range(tests):
#     print(db)
    errmean  = 0
    keycount = np.zeros(len(key))
    dball = initdball()
    for i in range(Iters):
        [dbraw, bad_entry] = explore()
        db = prepare(dbraw, bad_entry)
        [dbtrain, dbtest] = randsplitnorm(db, norm)
        [F, R] = auto(dbtrain, dbtest, xinp, unk, order, norm)
        dberr  = error(dbtest, F, R, norm, mpgmean, mpgstd)
        dbtop  = topcombo(dberr, topN)
        errmean  = errmean + np.mean(dbtop['err'])
        keycount = keycount + counthits(dbtop)
        dball = bestcombo(dbtop['fts'][0], dbtop['err'][0], dball)
    errmean  = errmean/(Iters)
    keycount = keycount/(topN*Iters)
    dbmax = findball(dball)
    print_results(order, Iters, topN, norm, errmean, keycount)
    print_max(order, Iters, topN, norm, dbmax)
