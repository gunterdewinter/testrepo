# Making the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import random
np.random.seed(0)

def printnetwork(layers):
    print('\nNEW')

    for layer in layers:
        #print(type(layer).__name__)
        if type(layer).__name__ == "FCLayer":
            print("weights")
            print(layer.weights)
        if type(layer).__name__ == "ActivationLayer":
            print("neuron")
            print(layer.neuron)
            


# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input, modify= True, usebias = True):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
    
#from layer import Layer
import numpy as np

# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        #self.bias = 0
        #input(f"rand weights {self.weights}")

        
 
    # returns output for a given input
    def forward_propagation(self, input_data, modify=True, usebias= True):
        #print(f"\nforward FC layer")
        self.input = input_data
        if usebias == False: self.bias = 0
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        #print(f"\nbackprop FC layer")

        input_error = np.dot(output_error, self.weights.T)
         
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        #input(f"weights {self.weights}")
        self.bias -= learning_rate * output_error
        #self.bias = 0
 
        return input_error
    
# inherit from base class Layer
class ActivationLayer(Layer):   

    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        self.neuron = 0

    # returns the activated input
    def forward_propagation(self, input_data, modify = True, usebias = True):
        #print(f"\nforward activation layer")
        self.input = input_data
 
        self.output = self.activation(self.input)
        if modify : self.neuron = self.output
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
       # print(f"\nbackprop activation layer")
        return self.activation_prime(self.input) * output_error
    

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)
        

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, x_test, y_test, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)
        errtest_list = []
        # training loop
        for i in range(epochs):
            err = 0
            err_test = 0
            for j in range(samples):  
                #print('\n')
                # forward propagation
                output = x_train[j]
                if j < len(x_test):
                    output_test = x_test[j]
                #input(f"input {x_train[j]}")
                #input(f"Train : {output}")
                for layer in self.layers:
                    #if type(layer).__name__ == "FCLayer":
                    #    print(f"Weights {layer.weights}")
                    #else: print(f"Neurons {layer.neuron}")    
                    if j < len(x_test):
                        output_test = layer.forward_propagation(output_test, True, True)
                    output = layer.forward_propagation(output, True, True)
                    #input(f"input {output}")
                    #input(f"Output {output}")    
                # compute loss (for display purpose only)
                #print("\nFF\n*************************************************************************************************\n")    
                #printnetwork(self.layers)
                
                err += self.loss(y_train[j], output)
                if j < len(x_test):
                    err_test += self.loss(y_test[j], output_test)
                    errtest_list.append(err_test)
                # backward propagation
                error = self.new_method(y_train, j, output)
                #input(f"\noutput error {error}")
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
    
                    #input(f"error {error}")
            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f    error_test=%f'   % (i+1, epochs, err, err_test))
        #plt.plot(errtest_list)
        #plt.show()

    def new_method(self, y_train, j, output):
        error = self.loss_prime(y_train[j], output)
        return error

def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;

def sigmoid(x): 
    return 1 / (1 + np.e**-x)

def sigmoid_prime(x):
    s=1/(1+np.exp(-x))
    ds=s*(1-s)  
    return ds

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;

def ReLU(x):
    return np.maximum(0.,x)

def ReLU_prime(x):   
    return np.greater(x, 0.).astype(np.float32)

def identity(x):   
    return x

def identity_prime(x):   
    return 1


