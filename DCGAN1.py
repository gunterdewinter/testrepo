#https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931

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



plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing Input data
data = pd.read_csv('train_lin.csv', delimiter = ';')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

x_corr = X.values.tolist()
y_corr = Y.values.tolist()

data_test = pd.read_csv('test_lin.csv', delimiter = ';')
X_test = data_test.iloc[:, 0]
Y_test = data_test.iloc[:, 1]


res = stats.pearsonr(x_corr, y_corr)



# manual rule based
if 1 > 2:
    m = random.randint(0,5) #0.15
    c =  random.randint(0,100) #70


    #plt.title(f"manual rule based  err: {err:.2f}  m: {m}  c: {c} corr: {round(res[0],2)}")
    #plt.scatter(X, Y)
    #plt.plot(X, Y_pred, color='red')
    #plt.show()

    epochs = 5000
    lr = 0.01
    d_range = 0.001

    for i in range(epochs):    
        Y_pred = m*X + c    
        err = np.mean(abs(Y-Y_pred))
        m_d=m
        c_d=c
    
        Y_pred_d = (m + d_range)*X + c
        err_temp = np.mean(abs(Y - Y_pred_d))
        err_d = err_temp / err
        if err_d > 1: 
            m_d -= lr * err_d
        else: 
            m_d += lr * err_d

        Y_pred_d = m*X + c + d_range
        err_d = np.mean(abs(Y - Y_pred_d)) / err
        if err_d > 1: c_d -= lr * err_d
        else: 
            c_d += lr * err_d

        m = m_d
        c = c_d
    

        print(f"epoch: {i} err: {err:.2f} m: {m:.2f}  c: {c:.2f}")
    
    plt.title(f"manual rule based  err: {round(err,2)} m: {m}  c: {c} corr: {round(res[0],2)}")
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()

if 1 > 2:
    # statistical rule based
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    num = 0
    den = 0
    for i in range(len(X)):
        num += (X[i] - X_mean)*(Y[i] - Y_mean)
        den += (X[i] - X_mean)**2
    m = num/den
    c = Y_mean - (m * X_mean)

    Y_pred = m*X + c

    err = np.mean(np.power(Y-Y_pred, 2))

    plt.title(f"stat rule based err: {round(err,2)} m: {round(m,2)}  c: {round(c,2)} corr: {round(res[0],2)}")
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()

if 1 > 0:
    # grad descent perceptron
    m = 0
    c = 0

    L = 0.0001  # The learning Rate
    epochs = 100000  # The number of iterations to perform gradient descent

    n = float(len(X)) # Number of elements in X

    # Performing Gradient Descent 


    m_list = []
    c_list = []

    for i in range(epochs): 
        Y_pred = m*X + c  # The current predicted value of Y
        #input(Y_pred)
        D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
        #input(D_m)
        D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
        m = m - L * D_m  # Update m
        c = c - L * D_c  # Update c
        print(f"{i} {m:.2f}\t{c:.2f}")
        c_list.append(c)
        m_list.append(m)
    
    #plt.plot(m_list)  

    #plt.show()
    print (m, c)

    # Making predictions
    Y_pred = m*X + c

    err = np.mean(np.power(Y-Y_pred, 2))

    plt.title(f"Perceptron based err: {round(err,2)}m: {round(m,2)}  c: {round(c,2)} corr: {round(res[0],2)}")
    plt.scatter(X, Y)
    plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
    plt.show()


if 1> 2:
    #data = pd.read_csv('data4.csv', delimiter = ';')
    #X = data.iloc[:, 0:2]
    #Y = data.iloc[:, 2]

    #X = data.iloc[:, 0]
    #Y = data.iloc[:, 1]


    x_train = X.to_numpy()  
    y_train = Y.to_numpy() 

    x_train = x_train.reshape(len(x_train),1,1)
    x_train = x_train.astype('float32')
    max_x = x_train.max()
    x_train /= max_x
    y_train = y_train.reshape(len(y_train),1,1)
    y_train = y_train.astype('float32')
    max_y = y_train.max()
    y_train /= max_y


    x_test = X_test.to_numpy()  
    y_test = Y_test.to_numpy() 

    x_test = x_test.reshape(len(x_test),1,1)
    x_test = x_test.astype('float32')
    max_x = x_test.max()
    x_test /= max_x
    y_test = y_test.reshape(len(y_test),1,1)
    y_test = y_test.astype('float32')
    max_y = y_test.max()
    y_test /= max_y




    print(x_train)
    print(y_train)

    hiddenlayer_neurons = 10


    net = Network()
    net.add(FCLayer(1, hiddenlayer_neurons))
    net.add(ActivationLayer(ReLU, ReLU_prime))
    #net.add(ActivationLayer(sigmoid, sigmoid_prime))
    ##net.add(ActivationLayer(identity, identity_prime))
    #net.add(ActivationLayer(tanh, tanh_prime))

    net.add(FCLayer(hiddenlayer_neurons, hiddenlayer_neurons))
    net.add(ActivationLayer(ReLU, ReLU_prime))
    #net.add(ActivationLayer(identity, identity_prime))
    #net.add(ActivationLayer(sigmoid, sigmoid_prime))
    #net.add(ActivationLayer(tanh, tanh_prime))

    #net.add(FCLayer(hiddenlayer_neurons, hiddenlayer_neurons))
    #net.add(ActivationLayer(ReLU, ReLU_prime))
    #net.add(ActivationLayer(identity, identity_prime))
    #net.add(ActivationLayer(sigmoid, sigmoid_prime))
    #net.add(ActivationLayer(tanh, tanh_prime))



    net.add(FCLayer(hiddenlayer_neurons, 1))
    #net.add(ActivationLayer(ReLU, ReLU_prime))
    #net.add(ActivationLayer(sigmoid, sigmoid_prime))



    # train
    net.use(mse, mse_prime)
    net.fit(x_train, y_train, x_test, y_test, epochs=5000, learning_rate=0.05)

    # test

    out = net.predict(x_train)
    out_test = net.predict(x_test)


    for i in range(0,len(x_train)):
        print(f" {round(float(x_train[i][0][0]*max_x),2)}  {round(float(out[i][0][0]*max_y),2)}")

    out_list = []
    out_test_list = []
    for i in range(0,len(out)):
        out_list.append(float(out[i][0][0]*max_y))
 
    for i in range(0,len(out_test)):
        out_test_list.append(float(out_test[i][0][0]*max_y))
    
    err = np.mean(np.power(Y-out_list, 2))

    plt.title(f"Dense NN based err: {round(err,2)}")
    plt.scatter(X, Y)
    plt.scatter(X, out_list)
    #plt.scatter(X_test, out_test_list)

    #plt.plot([min(X), max(X)], [min(out_list), max(out_list)], color='green')  # regression line
    plt.show()
