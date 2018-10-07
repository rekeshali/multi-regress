#!/usr/bin/python
import os.path
from explore import key

def print_results(iters, errmean, keycount):
    ofname = 'results.out'
    if os.path.isfile(ofname):
        of = open(ofname, 'a')
    else:
        of = open(ofname, 'w')
        line = '%5s %8s' % ('iters', 'errmean')
        for tooth in key[1:-1]:
            line = line + ' %12s' % (tooth)
        line = line + '\n'
        of.write(line)
    line = '%5i %8.5f' % (iters, errmean*100)
    for tooth in key[1:-1]:
        line = line + ' %12.2f' % (keycount[key.index(tooth)]*100)
    line = line + '\n'
    of.write(line)
