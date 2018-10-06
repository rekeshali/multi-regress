#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as mp

from explore import car, key
from prepare import db, dbst, keyst, dbtest, dbtrain, rand_idx, stat
from implement import build_X, build_R, linear_train, solve
from reduct import scatter 
from tools import autotrain, autotest

def auto(xidx, ridx):
# automatically trains and tests all possible combinations of data
    kidx = xidx
    ukidx = ridx
    W = autotrain(dbtrain, kidx, ukidx)
    R =  autotest(dbtest,  kidx, W[0])
    return W[1], W[0], R

def error(F, R):
# builds a database that holds all combos vs error stats
    Rtrue = build_R(dbtest, key[0])
    lenf = len(F)
    db = {}
    db['fts'] = []
    db['err'] = []
    for fidx in range(lenf):
        db['fts'].append(F[fidx])
        E = np.abs(R[fidx] - Rtrue)/Rtrue
        Emean = np.mean(E)
        Emin  = np.min(E)
        Emax  = np.max(E)
        Estd  = np.std(E)
        db['err'].append([Emean, Emin, Emax, Estd])
    return db

def topcombo(dberr, topn):
# extracts top n combinations with lowest mean error
    lendb = len(dberr['fts'])
    means = []
    for idx in range(lendb):
        means.append(dberr['err'][idx][0])
    midxs = np.argsort(means)
    db = {}
    db['fts'] = []
    db['err'] = []
    for idx in range(topn):
        db['fts'].append(dberr['fts'][midxs[idx]])
        db['err'].append(means[midxs[idx]])
    return db

S = auto([1,2,3,4,5,6,7],0)
dbfin = error(S[0], S[2])
top = topcombo(dbfin, 5)
