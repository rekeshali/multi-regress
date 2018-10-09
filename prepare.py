#!/usr/bin/python
import numpy as np
from explore import key

# macros for getting stats
def mean(db, tooth, numvals):
    return np.mean(db[tooth][:numvals])

def std(db, tooth, numvals):
    return np.std(db[tooth][:numvals])

def min(db, tooth, numvals):
    return np.min(db[tooth][:numvals])

def max(db, tooth, numvals):
    return np.max(db[tooth][:numvals])

def quartiles(db, tooth, numvals):
    return np.percentile(db[tooth][:numvals], [25, 50, 75])

def numvals(db, tooth):
    numvals = 0
    for num in db[tooth]:
        numvals = numvals + (num > -1)
    return numvals

# creating new key that sees discrete values 
from explore import db
disckeys = ['cylinders', 'model year', 'origin']
keylong = []
for tooth in key:
    if tooth not in disckeys:
        keylong.append(tooth)
    else:
        discvalues = []
        for val in db[tooth]:
            if val not in discvalues:
                discvalues.append(val)
        discvalues.sort()
        for val in discvalues:
            disctooth = tooth + str(val)[:-2]
            keylong.append(disctooth)

def stats(db, key):
    # create dict that holds statistics of data
    dbst = {}
    keyst = ['mean', 'std', 'min', 'max', 'quartiles', 'numvals']
    for tooth in key[:-1]:
        dbst[tooth] = {}
        for tooth1 in keyst:
            dbst[tooth][tooth1] = []

    # populate dict with stats
    for tooth in key[:-1]:
        dbst[tooth]['numvals'] = numvals(db, tooth)
        for tooth1 in keyst[:-1]:
            command = 'val = ' + tooth1 + '( db, "' + tooth + '", ' + str(dbst[tooth]['numvals']) + ')'
            exec(command)
            dbst[tooth][tooth1].append(val)
    return dbst

def znorm(db, dbst, key, discnorm):
    from explore import key as keyshort
    if discnorm:
        for tooth in key[:-1]:
            for i in range(len(db[key[0]])):
                db[tooth][i] = (db[tooth][i] - dbst[tooth]['mean'][0])/dbst[tooth]['std'][0]
    else:
        for tooth in key[:-1]:
            if tooth in keyshort:
                for i in range(len(db[key[0]])):
                    db[tooth][i] = (db[tooth][i] - dbst[tooth]['mean'][0])/dbst[tooth]['std'][0]
    return db

def prepare():
# imputate databse and do some stats
    from explore import db, bad_entry
    lbad = {}
    samples = len(db['mpg'])
    # relocating all bad entries to end
    for badkey in bad_entry:
        for badidx in bad_entry[badkey]:
          for tooth in key:
              db[tooth].append(db[tooth][badidx])
              db[tooth].pop(badidx)
        lbad[badkey] = len(bad_entry[badkey]) # for list navigation

#     # create dict that holds statistics of data
#     dbst = {}
#     keyst = ['mean', 'std', 'min', 'max', 'quartiles', 'numvals']
#     for tooth in key[:-1]:
#         dbst[tooth] = {}
#         for tooth1 in keyst:
#             dbst[tooth][tooth1] = []
# 
#     # populate dict with stats
#     for tooth in key[:-1]:
#         dbst[tooth]['numvals'] = numvals(db, tooth)
#         for tooth1 in keyst[:-1]:
#             command = 'val = ' + tooth1 + '( db, "' + tooth + '", ' + str(dbst[tooth]['numvals']) + ')'
#             exec(command)
#             dbst[tooth][tooth1].append(val)
    # do stats to get means
    dbst = stats(db, key)

    # imputate missing values with mean
    for badkey in bad_entry:
        for badidx in range(-1, -lbad[badkey] - 1, -1):
            db[badkey][badidx] = dbst[badkey]['mean'][0]

   # creating database with continuous and discrete data types
    dbdisc = {}
    for tooth in keylong:
        dbdisc[tooth] = []
        if tooth in key:
            dbdisc[tooth] = db[tooth]
        else:
            for disctooth in disckeys:
                lentooth = len(disctooth)
                if tooth[0] == disctooth[0]:
                    for val in db[disctooth]:
                        if int(tooth[lentooth:]) == val:
                            dbdisc[tooth].append(1)
                        else:
                            dbdisc[tooth].append(0)

#     [dbst, keyst] = stats(dbdisc, keylong)

    # normalizing
#     if norm == 1:
#         for tooth in key[:-1]:
#             if tooth not in disckeys:
#                 for i in range(samples):
#                     dbdisc[tooth][i] = (dbdisc[tooth][i] - dbst[tooth]['mean'][0])/dbst[tooth]['std'][0]
#     if norm == 1:
#         dbdisc = znorm(dbdisc, dbst, keylong)
#         for tooth in keylong[:-1]:
#             for i in range(samples):
#                 dbdisc[tooth][i] = (dbdisc[tooth][i] - dbst[tooth]['mean'][0])/dbst[tooth]['std'][0]

    return dbdisc


def randsplit(db):
# randomize and split data 
    import random
    num_samples = len(db[key[0]])
    num_train   = int(num_samples*0.75)
    num_test    = num_samples - num_train - 1

    rand_idx    = range(num_samples)
    random.shuffle(rand_idx)

    dbtrain = {}
    dbtest  = {}
    for tooth in keylong:
        i = 0
        dbtrain[tooth] = []
        dbtest[tooth]  = []
        for idx in rand_idx:
            if i <= num_train:
                dbtrain[tooth].append(db[tooth][idx])
            else:
                dbtest[tooth].append(db[tooth][idx])
            i = i + 1
    return dbtrain, dbtest

def randsplitnorm(db, norm):
    if norm == 'no':
        [dbtrain, dbtest] = randsplit(db)
        return dbtrain, dbtest
    if norm == 'sep':
        [dbtrain, dbtest] = randsplit(db)
        dbst    = stats(dbtrain, keylong)
        dbtrain = znorm(dbtrain, dbst, keylong, 0)
        dbst    = stats(dbtest, keylong)
        dbtest  = znorm(dbtest,  dbst, keylong, 0)
        return dbtrain, dbtest
    if norm == 'tog':
        dbst = stats(db, keylong)
        db   = znorm(db, dbst, keylong, 0)
        [dbtrain, dbtest] = randsplit(db)
        return dbtrain, dbtest
    if norm == 'dsep':
        [dbtrain, dbtest] = randsplit(db)
        dbst    = stats(dbtrain, keylong)
        dbtrain = znorm(dbtrain, dbst, keylong, 1)
        dbst    = stats(dbtest, keylong)
        dbtest  = znorm(dbtest,  dbst, keylong, 1)
        return dbtrain, dbtest
    if norm == 'dtog':
        dbst = stats(db, keylong)
        db   = znorm(db, dbst, keylong, 1)
        [dbtrain, dbtest] = randsplit(db)
        return dbtrain, dbtest

# quick acces to stats
def seestats(idx):
    if isinstance(idx, str): # if given name instead of index get index
        idx = key.index(idx)

    print('parameter: ' + key[idx]) # print key:value pairs
    for validx in range(0, len(keyst)):
        print(keyst[validx] + ': ' + str(dbst[key[idx]][keyst[validx]]))
