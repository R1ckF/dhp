import numpy as np

class relu:
    def forward_pass(self, x):
        return np.maximum(x, np.zeros_like(x))
    def diff(self,x):
        return x * (x>0)

class sigmoid:
    def forward_pass(self, x):
        return 1/(1+np.exp(-x))
    def diff(self, x):
        return self.forward_pass(x) * (1- self.forward_pass(x))

class softmax:
    def forward_pass(self, x):
        return np.exp(x)/(np.sum(np.exp(x)))
    def diff(self, x):
        return (np.exp(x) * np.sum(np.exp(x)) - (np.exp(x)**2)) / (np.sum(np.exp(x))**2)

class tanh:
    def forward_pass(self,x):
        return np.tanh(x)
    def diff(self,x):
        return 1- np.tanh(x)**2

class crossentropy:
    def forward_pass(self, x):
        return -1 * np.sum(target * np.log10(x) + (1-target) * np.log10(1-x))
    def diff(self, x):
        return -1 * ((target * 1/x) + (1-target) * (1/(1-x)))


class sum_square:
    def forward_pass(self, target, x):
        return np.sum(0.5*(target - x)**2)
    def diff(self, target, x):
        return x-target
