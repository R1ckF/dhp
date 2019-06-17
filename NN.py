import numpy as np
from activations import *
import time

class NN:
    """
    Class providing ANN structure includes forward_pass, backward_pass, and update
    Inputs:
        n_inputs: number of input nodes (int)
        n_outputs: number of output nodes (int)
        n_nodes: list of hidden notes, each entry represents a layer (list of int)
        lr: learning rate used
        activation: activaton used after each hidden layer
        weights/bias:    None use np.random.random as weight initializer
                        'zero' use np.zeros as weight initializer
                        'one' use np.ones as weight initializer
                        float use np.random.normal as weight initializer with sigma = inputs

    """
    def __init__(self, n_inputs, n_outputs, n_nodes=[2], lr=0.05, activation=relu,
            weights=None, bias=None):

        self.lr = lr
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = activation()
        self.forward_counter = 0
        self.backward_counter = 0
        self.update_counter = 0
        # self.history = {}

        def W_gen(input):
            #create weight initializer
            if input == None:
                return np.random.random
            elif input == 'zero':
                return np.zeros
            elif input == 'one':
                return np.ones
            else:
                def generator(size):
                    return np.random.normal(loc=0, scale=input, size=size)
                return generator

        def B_gen(input):
            #create bias_initializer
            if input == None:
                return np.random.random
            elif input == 'zero':
                return np.zeros
            elif input == 'one':
                return np.ones
            else:
                def generator(size):
                    return np.random.normal(loc=0, scale=input, size=size)
                return generator

        weights_gen = W_gen(weights)
        bias_gen = B_gen(bias)

        # initialize weights using initializer created above
        self.W = [weights_gen((n_inputs, n_nodes[0])).astype(np.float32)]
        for i in range(len(n_nodes)-1):
            self.W.append(weights_gen((n_nodes[i], n_nodes[i+1])).astype(np.float32))
        self.W.append(weights_gen((n_nodes[-1],n_outputs)).astype(np.float32))
        #initialize bias using initializer created above
        self.B = [bias_gen((n_node)).astype(np.float32) for n_node in n_nodes]
        self.B.append(bias_gen((n_outputs)).astype(np.float32))


    def forward_pass(self, x):
        # execute forward pass for given input x, x should be the same size as n_inputs indicate with dim(2)
        self.nodes_in = [x]
        self.nodes_out = [x]
        for w,b in zip(self.W[:-1], self.B[:-1]):
            h0_in = x @ w + b
            self.nodes_in.append(h0_in)
            x = self.activation.forward_pass(h0_in)
            self.nodes_out.append(x)

        h0_in = x @ self.W[-1] + self.B[-1]
        x = h0_in

        self.forward_counter += 1
        return x

    def dout_din(self):
        #output derivate of output with respect to input as required by DHP
        din = np.array([[1]])@self.W[-1].T
        for w, node_in in zip(reversed(self.W[:-1]), reversed(self.nodes_in[1:])):
            dout = din * self.activation.diff(node_in)
            din = dout @ w.T
        return din

    def backward_pass(self, dout):
        # obtain gradients for each weight and bias which can be used for an update_network
        # dout is the total gradient up to the end of the NN
        # the total gradient for each weight is calculated starting from the output nodes combining them up to the input nodes

        self.dW = []
        self.dB = []

        dhidden_in = dout

        for w, b, node_in, node_out in zip(reversed(self.W),reversed(self.B),reversed(self.nodes_in), reversed(self.nodes_out)):
            self.dB.append(np.multiply(dhidden_in, np.ones_like(b)))
            self.dW.append(np.outer(node_out,dhidden_in))

            dhidden_out =  dhidden_in @ w.T
            dhidden_in = dhidden_out * self.activation.diff(node_in)

        self.dW.reverse(), self.dB.reverse()
        self.backward_counter += 1
        return self.dW, self.dB, dhidden_out

    def update_network(self, dout=None):
        # the network is updated. If no backward_pass is completed yet it will first be executed.
        # If dout is not given and no backward_pass is previously executed an error will be thrown
        if self.backward_counter != self.update_counter +1 and dout==None:
            raise AssertionError
        elif self.backward_counter != self.update_counter +2 and dout!=None:
            _,_,_ = self.backward_pass(dout)
        else:

            W,B = [],[]
            for w,b,dw,db in zip(self.W,self.B,self.dW,self.dB):
                W.append(w-self.lr*dw)
                B.append(b-self.lr*db)
            self.W, self.B = W,B
            self.update_counter += 1
