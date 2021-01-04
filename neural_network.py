'''
# Name: Daniel Johnson
# Made With: Grant Holmes, https://github.com/gholmes829
# File: neural_network.py
# Date: 1/3/2021
# Brief: This file provides a neural network implementation that can be
#        trained using back propagation and gradient descent
'''

import numpy as np
from costs import *
from activations import *

class Neural_Network:
    '''
    # @post: A Neural_Network object is created
    # @param: architecture: a tuple to represent the layers and length of layers
    #         activation: activation function
    #         aGrad: gradient of activation function
    #         cost: cost function
    #         cGrad: gradient of cost function
    #         weights: list of numpy arrays giving weight values, randomly selects if none is given
    #         biases: list of bias values, randomly selects if none is given
    '''
    def __init__(self, architecture: tuple, activation: str = "sigmoid", aGrad: str = "sigmoid", cost: str = "CrossEntropy", cGrad: str = "CrossEntropy", weights: list = None, biases: list = None) -> None:
        #initialize dictionaries that link input strings to functions
        activations = {
            "sigmoid": sigmoid,
            "tanh": tanh
        }
        actGradients = {
            "sigmoid": sigmoidGradient,
            "tanh": tanhGradient
        }
        costs = {
            "MSE": MeanSqErr,
            "CrossEntropy": CrossEntropy
        }
        costGradients = {
            "MSE": MeanSqErrGradient,
            "CrossEntropy": CrossEntropyGradient
        }

        #architecture is the length of each layer in a tuple ex: (2,3,1)
        self.architecture = architecture
        self.layers = len(architecture)

        #make a list containing the shapes of each layer that should have weights
        weightShapes = [(i, j) for i, j in zip(architecture[1:], architecture[:-1])]
        
        #let the user set weights and biase s manually, else randomly initialize
        self.weights = [np.random.randn(*s) for s in weightShapes] if weights is None else weights
        self.biases = [np.random.standard_normal(s) for s in architecture[1:]] if biases is None else biases
        
        #use the dictionaries to link functions to member variables
        self.activation = activations[activation]
        self.aGradient = actGradients[aGrad]
        self.cost = costs[cost]
        self.cGradient = costGradients[cGrad]

    '''
    # @param: X: a numpy array designating input values to the network
    # @return: a list of predictions based on X
    '''
    def forwardProp(self, X: np.ndarray) -> np.ndarray:
        #for each layer that has weights and biases, 
        #compute a = XTheta + b
        feed = X
        for w, b in zip(self.weights, self.biases):
            feed = self.activation( np.dot(feed, w.T) + b )
        return feed

    '''
    # @param: X: a numpy array designating input values to the network
    #         y: a numpy array designating given output values 
    # @return: two lists: weight gradient: dC/dw, and bias gradient: dC/db  
    '''
    def backwardProp(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        m = len(X)
        aValues, zValues, deltas = [], [], []
        aValues.append(X) #the imput layer does not activate

        #preform forward propagation and save all intermediate a and z values
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):		
            z = np.dot(aValues[i], w.T) + b
            zValues.append(z)
            aValues.append(self.activation(z))

        #assign final delta manually
        deltas.append( self.cGradient(y, aValues[-1]) * self.aGradient(zValues[-1]) )
        for i in range(self.layers - 2):
            #backpropagate through each layer computing the delta-i+1 = delta-i @ w-(-i-1) * sigDer(z-i)
            error = ( np.dot(deltas[i], self.weights[-i - 1]) ) * sigmoidGradient(zValues[i])
            deltas.append(error)
        #we appended forward, so the list needs to be reversed
        deltas.reverse() 

        weightGradient, biasGradient = [], []
        #loop through each layer, compute and store the weight and bias gradients, and return them
        for i in range(self.layers - 1):
            weightGradient.append( (1/m) * np.dot(deltas[i].T, aValues[i]) )
            biasGradient.append( (1/m) * np.dot(deltas[i].T, np.ones(m)) )
        return weightGradient, biasGradient

    '''
    # @post: self.weights and self.biases are updated from gradient descent
    # @param: X_train: the input training values
    #         y_train: the output training values
    #         alpha: learning rate, has default value
    #         maxIter: maximum number of training iterations, has default value
    #         convergenceThreshold: The minimum cost improvement before the training
    #                               loop will break. Does not break by default
    # @return: a list containing the cost at every iteration
    '''
    def train(self, X_train: np.ndarray, y: np.ndarray, alpha: float = 0.1, maxIter: int = 1000, convergenceThreshold: float = 0.0) -> list:
        J_History = []
        #the first cost is before any learning takes place
        J_History.append(self.cost(y, self.forwardProp(X_train)))
        for it in range(maxIter):
            #get the gradients based on current weights and biases, and perform gradient descent
            weightGradient, biasGradient = self.backwardProp(X_train, y)

            #for each weight and bias, "step down the hill"
            #since this is regular gradient descent, we can get stuck in local optima  
            self.weights = [self.weights[i] - alpha * weightGradient[i] for i in range(len(self.weights))]
            self.biases = [self.biases[i] - alpha * biasGradient[i] for i in range(len(self.biases))]
            
            #append the cost and print info
            J_History.append(self.cost(y, self.forwardProp(X_train)))
            print("Iteration: ", it, "Cost: ", J_History[it])

            #if we have converged to the convergence threshold, break
            if (J_History[it - 1] - J_History[it] < convergenceThreshold and it > 0):
                print("Training converged at iteration:", it)
                break
        return J_History

    '''
    # @post: All of the weights and biases are printed to screen in a nice fashion
    '''
    def disp(self):
        #loop through all of the weights and biases and print them rounded to 2 decimal places
        for i in range(self.layers - 1):
            print("Layer:", i + 1, "-", i + 2)
            for j in range(len(self.weights[i])):
                print(f"w{j}:", (np.round(self.weights[i][j], 2)), "   ", end="") #end="" is for no \n
            print("bias: ", round(self.biases[i][0], 2))