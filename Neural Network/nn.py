import numpy as np
from layers import *
from errors import *
from activations import *

class NN(object):
  """
  Define the structure
  Inputs : input_dim = Number of features in the input data (for image, flatten the input),
            hidden_dim = List of integers for number of hidden neurons in each dimension,
            n_classes = Number of classes in the output
  
  Outputs : undefined
  """
  def __init__(self, input_dim, hidden_dims, n_classes, loss_type, lr = 1e-3, batch_size = 50, n_epoch = 10, reg = 1e-3, weight_scale = 1e-3):
    self.lr = lr
    self.batch_size = batch_size
    self.epochs = n_epoch
    self.n_classes = n_classes
    self.reg = reg
    self.n_layers = len(hidden_dims) + 1
    self.loss_type = loss_type
    
    self.params = {}
    
    # Initialize Weights for the hidden layers from normal distribution
    for i in range(self.n_layers - 1):
      self.params['w' + str(i+1)] = np.random.normal(0, weight_scale, [input_dim, hidden_dims[i]])
      self.params['b' + str(i+1)] = np.zeros(hidden_dims[i])
      
      input_dim = hidden_dims[i]
      
    # Initialize Weights for the output layer
    self.params['w' + str(self.n_layers)] = np.random.normal(0, weight_scale, [hidden_dims[self.n_layers - 2], n_classes])
    self.params['b' + str(self.n_layers)] = np.zeros(n_classes)
    
    
  def fit(self, X, labels):
    # Function used to train the weights
    pass
  
  def predict(self, X, y):
    # Function used to predict the scores for unknown input
    pass
        
  def forward_pass(self, X, labels=None):
    
    scores = None
    mode = 'TEST' if labels is None else 'TRAIN'
    fc_cache = {} # Cache the inputs to hidden layers for use in back prop
    relu_cache = {} # Cache the inputs to ReLU layers for use in back prop
    #(self.n_layers)
    for i in range(self.n_layers - 1):
      fc_out, fc_cache[str(i + 1)] = affine_forward(X, self.params['w' + str(i + 1)], self.params['b' + str(i+1)])
      relu_out, relu_cache[str(i + 1)] = relu_forward(fc_out)
      X = fc_out
      
    # Logic for the output layer
    scores, fc_cache[str(self.n_layers)] = affine_forward(fc_out, self.params['w' + str(self.n_layers)], self. params['b' + str(self.n_layers)])
    
    # If there are no labels provided then we in test mode so return the scores
    if mode == 'TEST': 
      return scores, fc_cache, relu_cache
    
    # Else calculate the loss and perform backward propogation to fix the weights
    #loss, grads = calculate_loss(scores, fc_cache, relu_cache)
    
    '''
    _________TO DO_____________
    
    From the gradients calculated perform weight update
    '''
    return scores, fc_cache, relu_cache 
    pass
    
  def calculate_loss(self, scores, fc_cache, relu_cache):
    '''
    Calculate gradients and perform weight update
    '''
    loss, grads = 0.0, {}
    
    # Calculate loss and add it to the total loss.
    if self.loss_type == 'MSE':
      batch_loss, theta = MSE(scores, labels)
    loss += batch_loss
    
    # calculate gradients for the first layer and pass it downstream
    grads['w' + str(self.n_layers)] = theta*fc_cache[self.n_layers]
    for i in range(self.n_layers - 1, 0, -1):
      out, _ = affine_backward(grads['w' + str(i + 1)], fc_cache['i'])
      grads['x' + str(i)], grads['w' + str(i)], grads['b' + str(i)], grads = out
      
      
      pass
    pass
    
    
m = NN(4, [2,4,6,8], 2, 'MSE')
y = [0,1,1,1]
x = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])

print(x)
print(y)

scores, fc_cache, relu_cache = m.forward_pass(x,y)
m.calculate_loss(scores, fc_cache, relu_cache)
print(scores)
#for i in range(m.n_layers):
#  print(m.params['w' + str(i+1)].shape, '+', m.params['b' + str(i+1)].shape)
  


    
    
    
