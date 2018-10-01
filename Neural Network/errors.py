import numpy as np

def MSE(y, y_):
  # Predicted class score
  y = np.max(scores, axis = 1)
  
  # Target class score
  y_ = scores[np.arange(self.batch_size), np.where(y_ == 1)[1]]
  
  loss = (1/2)*(y - y_)**2
  theta = y_ - y
  return y, y_, loss

def CEL(y, y_):
  pass
  
#def 
  
  
  