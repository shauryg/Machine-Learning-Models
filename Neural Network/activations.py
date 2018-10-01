import numpy as np

def relu_forward(x):
  """
   Computes the forward pass for a layer of rectified linear units (ReLUs).
   Input:
   - x: Inputs, of any shape
   Returns a tuple of:
   - out: Output, of the same shape as x
   - cache: x
  """
  return np.maximum(x,0), x
 
def relu_backward(dout, cache): 
  """
   Computes the backward pass for a layer of rectified linear units (ReLUs).
   Input:
   - x: Inputs, of any shape
   Returns a tuple of:
   - out: Output, of the same shape as x
   - cache: x
  """
  dx, x = dout, cache
  dx = dout * np.where(x > 0, 1, 0)
  return dx
  pass
  
def sigmoid_forward(x):
  pass
  
def sigmoid_backward(dout, cache):
  pass