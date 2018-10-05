#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as mp
from prepare import db, key

def scatter(x, y):
    fig = mp.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x,y)
    mp.show()
