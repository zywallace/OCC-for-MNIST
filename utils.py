import numpy as np
import cPickle
import gzip
from copy import deepcopy

# Load data

def load_data():
    f = gzip.open('mnist.pkl.gz','rb')
    training_data,validation_data,test_data=cPickle.load(f)
    f.close()
    return training_data[0],training_data[1],validation_data[0],validation_data[1],test_data[0],test_data[1]

# one class
def oneClass_data(X, Y, target = 0):
    """
    given X and Y, divide sample into target class and others
    
    return:
    X from target class and X from other class
    """
    return X[Y == target], X[Y != target]

# superclass of modules
class Module:
    """
    Module is a super class. It could be a single layer, or a multilayer perceptron.
    """
    
    def __init__(self):
        self.train = True
        return
    
    def forward(self, _input):
        """
        h = f(z); z is the input, and h is the output.
        
        Inputs:
        _input: z
        
        Returns:
        output h
        """
        pass
    
    def backward(self, _input, _gradOutput):
        """
        Compute:
        gradient w.r.t. _input
        gradient w.r.t. trainable parameters
        
        Inputs:
        _input: z
        _gradOutput: dL/dh
        
        Returns:
        gradInput: dL/dz
        """
        pass
        
    def parameters(self):
        """
        Return the value of trainable parameters and its corresponding gradient (Used for grandient descent)
        
        Returns:
        params, gradParams
        """
        pass
    
    def training(self):
        """
        Turn the module into training mode.(Only useful for Dropout layer)
        Ignore it if you are not using Dropout.
        """
        self.train = True
        
    def evaluate(self):
        """
        Turn the module into evaluate mode.(Only useful for Dropout layer)
        Ignore it if you are not using Dropout.
        """
        self.train = False

def sgdmom(x, dx, lr, alpha = 0, state = None, weight_decay = 0):
    # sgd momentum, uses nesterov update (reference: http://cs231n.github.io/neural-networks-3/#sgd)
    if not state:
        if type(x) is list:
            state = [None] * len(x)
        else:
            state = {}
            state['m'] = np.zeros(x.shape)
            state['tmp'] = np.zeros(x.shape)
    if type(x) is list:
        for _x, _dx, _state in zip(x, dx, state):
            sgdmom(_x, _dx, lr, alpha, _state)
    else:
        state['tmp'] = state['m'].copy()
        state['m'] *= alpha
        state['m'] -= lr * (dx +  weight_decay * x)  
        
        x -= alpha * state['tmp']
        x += (1 + alpha) * state['m']
        
