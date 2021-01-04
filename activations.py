'''
# Name: Daniel Johnson
# File: activations.py
# Date: 1/3/2021
# Brief: this file contains activation functions
'''

import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))

def tanh(x):
    return np.tanh(x)
def tanhGradient(x):
    return 1-np.tanh(x)**2