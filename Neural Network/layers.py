import numpy as np

def affine_forward(x,w,b):
  """
  Computes the forward pass for an affine (fully-connected) layer.
  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.
  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  cache = (x, w, b)
  out = np.add(np.dot(x,w),b)
  return out, cache
      
def affine_backward(dout, cache):
  """
   Computes the backward pass for an affine layer.
   Inputs:
   - dout: Upstream derivative, of shape (N, M)
   - cache: Tuple of:
     - x: Input data, of shape (N, d_1, ... d_k)
     - w: Weights, of shape (D, M)
   Returns a tuple of:
   - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
   - dw: Gradient with respect to w, of shape (D, M)
   - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None
  
  dw = np.dot(dout, x)
  dx = np.dot(dout, w)
  out = dx, dw, db
  return out, cache
 
def batchnorm_forward(x, gamma, beta, bn_param):

  pass
  
def batchnorm_backward(dout, cache):

  pass
  
'''
def softmax(x):
""" Calculates the softmax of the output from """
stableX = x - np.array([np.max(x,axis=1)]).T
exp = np.exp(exp)
return exp/np.sum(exp, axis = 1)

def calculate_loss(z,y):
#z = final output from the forward pass through all the layers
z = np.exp(z)/np.sum(np.exp(z), axis=1)[:, np.newaxis]
#loss = [N,], where N = batch_size
loss = -np.log(z[np.arange(np.shape(prob)[0]),np.where(y==1)[1]])
'''
    
  