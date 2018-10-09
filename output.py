#!/usr/bin/python
import os.path
from explore import key

def print_results(order, iters, topN, norm, errmean, keycount):
    ofname = 'results.out'
    if os.path.isfile(ofname):
        of = open(ofname, 'a')
    else:
        of = open(ofname, 'w')
        line = '%5s %5s %5s %5s %8s' % ('order', 'iters', 'topN','norm', 'errmean')
        for tooth in key[1:-1]:
            line = line + ' %6s' % (tooth[:6])
        line = line + '\n'
        of.write(line)
    line = '%5i %5i %5i %5s %8.4f' % (order, iters, topN, norm, errmean*100)
    for tooth in key[1:-1]:
        line = line + ' %6.2f' % (keycount[key.index(tooth)]*100)
    line = line + '\n'
    of.write(line)

def print_max(order, iters, topN, norm, dbmax):
    ofname = 'max.out'
    if os.path.isfile(ofname):
        of = open(ofname, 'a')
    else:
        of = open(ofname, 'w')
        line = '%5s %5s %5s %5s %5s %8s %19s' % ('order', 'iters', 'topN', 'norm', 'count', 'errmean', 'feature combination') + '\n'
        of.write(line)
    line = '%5i %5i %5i %5s %5i %8.4f' % (order, iters, topN, norm, dbmax['count'], dbmax['err']*100)
    for feat in dbmax['features']:
        line = line + ' %s,' % (feat)
    line = line[:-1] + '\n'
    of.write(line)
