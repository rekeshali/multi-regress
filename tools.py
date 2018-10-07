#!/usr/bin/python
import numpy as np
from explore import key

def getkey(key, xinp):
    features = []
    for i in xinp:
        features.append(key[i])
    return features

def len2dec(length):
    decimal = 0
    for i in range(length):
        decimal += pow(2,i)
    return decimal

def dec2bin(decimal):
    binary = []
    if decimal == 0:
        binary += [0]
    while decimal > 0:
        binary  = [decimal%2] + binary
        decimal = decimal/2
    return binary

def bin2flag(binary, length):
    lenbin = len(binary)
    fidxstr = list(np.zeros(length-lenbin))
    fidxstr = fidxstr + list(binary)
    binint = []
    for lidx in range(length):
        binint.append(int(fidxstr[lidx]))
    return binint

def autotrain(dbtrain, xinp, ukidx):
    from explore import key
    from implement import linear_train
    ukey = key[ukidx]
    features = getkey(key, xinp)
    lenfeats = len(features)
    bindec = len2dec(lenfeats)
    F = []
    W = []
    for fidx in range(bindec):
        xidx = []
        binlist = dec2bin(fidx+1)
        flagidx = bin2flag(binlist,lenfeats)
        for i in range(lenfeats):
            if flagidx[i] == 1:
                xidx.append(i)
        S = linear_train(dbtrain, features, xidx, ukey)
        F.append(getkey(features,xidx))
        W.append(S[0])
    return W, F

def autotest(dbtest, xinp, W):
    from explore import key
    from implement import solve
    features = getkey(key, xinp)
    lenfeats = len(features)
    bindec = len2dec(lenfeats)
    R = []
    for fidx in range(bindec):
        xidx = []
        binlist = dec2bin(fidx+1)
        flagidx = bin2flag(binlist,lenfeats)
        for i in range(lenfeats):
            if flagidx[i] == 1:
                xidx.append(i)
        S = solve(dbtest, features, xidx, W[fidx])
        R.append(S[0])
    return R

def auto(dbtrain, dbtest, xidx, ridx):
# automatically trains and tests all possible combinations of data
    kidx = xidx
    ukidx = ridx
    [W, F] = autotrain(dbtrain, kidx, ukidx)
    R =  autotest(dbtest,  kidx, W)
    return F, R

def error(dbtest, F, R):
# builds a database that holds all combos vs error stats
    from implement import build_R
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

def counthits(dbtop):
    keycount = np.zeros(len(key))
    for tooth in dbtop['fts']:
        for tooth1 in tooth:
            kidx = key.index(tooth1)
            keycount[kidx] += 1
    return keycount
