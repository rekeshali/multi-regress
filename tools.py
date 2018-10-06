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

def autotrain(dbtrain, xinp, ukidx):
    from explore import key
    from implement import linear_train
    ukey = key[ukidx]
    features = getkey(key, xinp)
#     print(features)
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
#         print(S[0])
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






