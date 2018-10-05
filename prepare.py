#!/usr/bin/python

import numpy as np
from explore import key, db, bad_entry

# macros for getting stats
def mean(tooth, numvals):
    return np.mean(db[tooth][:numvals])

def std(tooth, numvals):
    return np.std(db[tooth][:numvals])

def min(tooth, numvals):
    return np.min(db[tooth][:numvals])

def max(tooth, numvals):
    return np.max(db[tooth][:numvals])

def quartiles(tooth, numvals):
    return np.percentile(db[tooth][:numvals], [25, 50, 75])

def numvals(tooth):
    numvals = 0
    for num in db[tooth]:
        numvals = numvals + (num > -1)
    return numvals

lbad = {}
# relocating all bad entries to end
for badkey in bad_entry:
    for badidx in bad_entry[badkey]:
      for tooth in key:
          db[tooth].append(db[tooth][badidx])
          db[tooth].pop(badidx)
    lbad[badkey] = len(bad_entry[badkey]) # for list navigation

# create dict that holds statistics of data
dbst = {}
keyst = ['mean', 'std', 'min', 'max', 'quartiles', 'numvals']
for tooth in key[:-1]:
    dbst[tooth] = {}
    for tooth1 in keyst:
        dbst[tooth][tooth1] = []

# populate dict with stats
for tooth in key[:-1]:
    dbst[tooth]['numvals'] = numvals(tooth)
    for tooth1 in keyst[:-1]:
        command = 'val = ' + tooth1 + '("' + tooth + '", ' + str(dbst[tooth]['numvals']) + ')'
        exec(command)
        dbst[tooth][tooth1].append(val)

# quick acces to stats
def stat(idx):
    if isinstance(idx, str): # if given name instead of index get index
        idx = key.index(idx)

    print('parameter: ' + key[idx]) # print key:value pairs
    for validx in range(0, len(keyst)):
        print(keyst[validx] + ': ' + str(dbst[key[idx]][keyst[validx]]))

# imputate missing values with mean
for badkey in bad_entry:
    for badidx in range(-1, -lbad[badkey] - 1, -1):
        db[badkey][badidx] = dbst[badkey]['mean'][0]

# recharacterize certain data types such as discretes


# randomize and split data 
import random
num_samples = len(db[key[0]])
num_train   = int(num_samples*0.75)
num_test    = num_samples - num_train - 1

rand_idx    = range(num_samples)
random.shuffle(rand_idx)

dbtrain = {}
dbtest  = {}
for tooth in key:
    i = 0
    dbtrain[tooth] = []
    dbtest[tooth]  = []
    for idx in rand_idx:
        if i <= num_train:
            dbtrain[tooth].append(db[tooth][idx])
        else:
            dbtest[tooth].append(db[tooth][idx])
        i = i + 1

