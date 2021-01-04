'''
# Name: Daniel Johnson
# File: main.py
# Date: 1/3/2021
# Brief: This script creates uses neural_network to train 
#        and evaluate the network using XOR as an example
'''

import numpy as np
import matplotlib.pyplot as plt
from neural_network import Neural_Network

def main():
    #make a neural network with set architecture
    arch = (2,4,1)
    nn = Neural_Network(arch)

    #XOR input data
    X_train = np.array( [ [0,0], [0,1], [1,0], [1,1] ] )
    #XOR output data
    y_train = np.array( [[0],[1],[1],[0]] )

    #set max iterations, learning rate, and convergence threshold
    iters, lr, threshold = 5000, 1, 0.00001
    #train the network
    J_Hist = nn.train(X_train, y_train, alpha = lr, maxIter = iters, convergenceThreshold = threshold)

    #forward propagate to get a prediction from the network
    result = nn.forwardProp(X_train)

    #print some nice information
    print("\nUnfiltered Prediction:\n", result)
    print("Final Prediction:\n", result >= 0.5, '\n')
    print("Random init cost: ", round(J_Hist[0], 5), ", Final cost: ", round(J_Hist[-1], 5))
    print("Cost reduction from random init: ", round(J_Hist[0] - J_Hist[-1], 5), '\n')

    #set up subplots for the cost history and decision boundary
    figure, plots = plt.subplots(ncols=2)
    figure.suptitle('Neural Network Learning of XOR') #supertitle
    figure.tight_layout(pad=2.5, w_pad=1.5, h_pad=0) #fix margins
    drawCostHistory(J_Hist, plots[0])
    drawDecisionBoundary(nn, plots[1], seperation_coefficient = 50, square_size = 1, allowNegatives = False)
    #show the cool graphs :)
    plt.show()

'''
# @param: J_Hist: a list of costs to plot over iterations
#         plot: a matplotlib plot
# @post: a curve is plotted showing cost over iterations
'''
def drawCostHistory(J_Hist: list, plot):
    plot.plot(J_Hist)
    plot.set_ylabel('Cost')
    plot.set_xlabel('Iterations')
    plot.set_title('Cost vs. Iterations')
    plot.axis([0, len(J_Hist), 0, max(J_Hist)])
    plot.set_aspect(len(J_Hist)/max(J_Hist))

'''
# @param: nn: a Neural_Network object
#         plot: a matplotlib plot
#         seperation_coefficient: a measure of how many points there are per 1 unit
#         square_size: length and width of area plotted
#         allowNegatives: if true, it will shift the data into four quadrants instead of just positive positive
# @post: points are plotted to the plot with color depending on their nn result
'''
def drawDecisionBoundary(nn, plot, seperation_coefficient: int = 50, square_size: int = 1, allowNegatives: bool = False):
    #create a 2d array of input values depending on the parameters
    X_Test = []
    for i in range(seperation_coefficient + 1):
        X_Test.append([])
        for j in range(seperation_coefficient + 1):
            xVal = i*square_size/seperation_coefficient
            yVal = j*square_size/seperation_coefficient
            if allowNegatives:
                xVal -= square_size/2.0
                yVal -= square_size/2.0
            X_Test[i].append([xVal, yVal])

    #get the results from the range of values and plot them
    test_result = nn.forwardProp(X_Test)
    for i in range(len(test_result)):
        for j in range(len(test_result)):
            xVal = X_Test[i][j][0]
            yVal = X_Test[i][j][1]
            clr = (1 - test_result[i][j][0],0,test_result[i][j][0])
            #this other line plots the >= .5 binary predictions instead of range of purples
            #clr = "blue" if test_result[i][j] >= 0.5 else "red"
            plot.plot(xVal, yVal, color=clr, marker="s")
    plot.set_ylabel('X2')
    plot.set_xlabel('X1')
    plot.set_title('Decision Boundary [X1, X2] -> Y\nBlue = 1, Red = 0')
    plot.set_aspect('equal')

if __name__ == "__main__": main()