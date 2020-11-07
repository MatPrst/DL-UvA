"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    
    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.
    
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
    
        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.
    
        Also, initialize gradients with zeros.
        """

        self.params = {}
        self.params["weight"] = np.random.normal(loc=0, scale=0.0001, size=(out_features, in_features))
        self.params["bias"] = np.zeros((1, out_features))

        self.grads = {}
        self.grads["weight"] = np.zeros((out_features, in_features))
        self.grads["bias"] = np.zeros((1, out_features))

        self.x = None
    
    def forward(self, x):
        """
        Forward pass.
    
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        self.x = x.copy()
        batch_size = x.shape[0]
        B = np.tile(self.params["bias"], (batch_size, 1))
        out = x @ self.params["weight"].T + B
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
    
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """
        
        self.grads["weight"] = dout.T @ self.x
        self.grads["bias"] = np.ones(dout.shape[0]) @ dout
        dx = dout @ self.params["weight"]
        
        return dx



class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def __init__(self):
        self.x = None
    

    def _softmax(x):
        """Return sofmax using Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        """

        b = np.max(x, axis=1)
        y = np.exp(x - b.reshape(-1, 1))
        out = y / np.sum(y, axis=1, keepdims=True)

        return out 
    
    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        self.x = x.copy()
        out = SoftMaxModule._softmax(x)
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module.
        """
        
        s = SoftMaxModule._softmax(self.x)

        # Softmax derivative
        common = np.einsum('ij,ik->ijk', s, s)
        n = self.x.shape[1]
        diag = np.einsum('ij,jk->ijk', s, np.eye(n, n))
        
        dx = np.einsum('in,ijn->ij', dout, diag - common)
        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """
    
    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
    
        TODO:
        Implement forward pass of the module.
        """

        Li = - np.sum(y*np.log(x), axis=1)
        out = np.mean(Li)
        
        return out
    
    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
    
        TODO:
        Implement backward pass of the module.
        """

        s = x.shape[0]
        dx = -1/s * y / x
        
        return dx


class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        self.x = x.copy()
        out = np.where(x >= 0, x, np.exp(x) - 1)
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        d = np.where(self.x >= 0, 1, np.exp(self.x))
        dx = np.multiply(dout, d)

        return dx