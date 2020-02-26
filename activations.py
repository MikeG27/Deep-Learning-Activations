import numpy as np

def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

def tanh(z):
    t = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    return t

def relu(z):
    return z * (z > 0)

def softmax(z):
    s = (np.exp(z)/np.sum(np.exp(z)))
    return s


