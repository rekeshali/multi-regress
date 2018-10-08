#!/usr/bin/python
import numpy as np

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

def autotrain(db, xinp, unkidx, order):
    # models polynomial of order for all possible combinations of xinp in db on unkidx in db
    from explore import key
    from implement import train
    unkey = key[unkidx]
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
        featcombo = getkey(features, xidx)
        W.append(train(db, featcombo, unkey, order))
        F.append(featcombo)
    return W, F

def autotest(db, xinp, order, W):
    # tests polynomial of order for all possible combinations of xinp in db on unkidx in db
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
        featcombo = getkey(features, xidx)
        R.append(solve(db, featcombo, order, W[fidx]))
    return R

def auto(dbtrain, dbtest, xidx, ridx, order):
    # automatically trains and tests all possible combinations of data
    [W, F] = autotrain(dbtrain, xidx, ridx, order)
    R =  autotest(dbtest, xidx, order, W)
    return F, R

def error(dbtest, F, R):
    # builds a database that holds all combos vs error stats
    from implement import buildR
    from explore import key
    Rtrue = buildR(dbtest, key[0])
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
    # counts number of times a keyword appears in the top list
    from explore import key
    keycount = np.zeros(len(key))
    for tooth in dbtop['fts']:
        for tooth1 in tooth:
            kidx = key.index(tooth1)
            keycount[kidx] += 1
    return keycount

def initdball():
    dball = {}
    dball['keybool'] = []
    dball['combocount'] = []
    dball['err'] = []
    return dball

def bestcombo(topfeats, toperr, dball, lol):
    # find #1 combination of features across all iterations
    from explore import key
    keybool = list(np.zeros(len(key)))
    for tooth in topfeats:
        keybool[key.index(tooth)] = 1.0
#     print(keybool)
    if len(dball['keybool']) == 0:
        dball['keybool'].append(keybool)
        dball['combocount'].append(1.0)
        dball['err'].append(toperr)
    else:
        for combo in dball['keybool']:
#             print('combo',combo)
            if combo == keybool:
                idx = dball['keybool'].index(combo)
                dball['combocount'][idx] += 1.0
                dball['err'][idx] += toperr
#                 print('eq')
                break
            else:
                dball['keybool'].append(keybool)
                dball['combocount'].append(1.0)
                dball['err'].append(toperr)
#                 print('neq')
                break
#     print('lol: ' + str(lol))
    lol += 1
#     print('lol',lol)
    return dball, lol

def findball(dball):
    from explore import key
    maxcount = np.max(dball['combocount'])
    maxidx = dball['combocount'].index(maxcount)
    maxbool = dball['keybool'][maxidx]
    maxerr = dball['err'][maxidx]/maxcount

    db = {}
    db['count'] = maxcount
    db['err'] = maxerr
    db['features'] = []
    for i in range(len(key)):
        if maxbool[i] == 1.0:
            db['features'].append(key[i])
    return db

