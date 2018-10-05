#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as mp

from explore import car
from prepare import db, key, dbst, keyst, dbtest, dbtrain, rand_idx, stat
# from implement import Xtrain, Xtest, Rtrain, Rtest, W
from implement import linear
from reduct import scatter 

inputs = key[1:-1]


