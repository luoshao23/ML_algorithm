from numpy import *
import numpy as np

def difcost(a,b):
    dif = sum(power(a-b,2))
    return dif
