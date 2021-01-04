'''
# Name: Daniel Johnson
# File: costs.py
# Date: 1/3/2021
# Brief: this file contains cost functions
'''

import numpy as np

def MeanSqErr(y, hx):
    return (0.5*np.sum(np.square(y - hx))) / y.shape[0]

def MeanSqErrGradient(y, hx):
    return 2*(hx - y)/y.size

def CrossEntropy(y, hx):
    return -(1/y.shape[0]) * np.sum(y * np.log(hx) + (1 - y) * np.log(1 - hx) )

def CrossEntropyGradient(y, hx):
    return -( (y/hx) - ( (1 - y) / (1 - hx) ) )