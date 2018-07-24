import numpy as np
class LogisticRegression(object):
  """Logistic Regression Classifier
  
  Parameters
  -----------
  eta : float
    Learning rate (between 0.0 and 1.0)
  n_iter : int
    Passes over training dataset
  random_state : int
    Random number generator to seed for random weight
  batch_size : int
    Number of batches to divide one epoch in
    
  Attributes
  ------------
  w_ : 1-d array
    weights for each independent input
  b_ : bias variable added to weight output
  cost_ : list
    logarithmic cost value in each epoch
  """
  def __init__(self, lr = 0.001, n_iter = 50, random_state = 1, batch_size = 10):
    
    self.lr = lr
    self.n_iter = n_iter
    self.random_state = random_state
    self.batch_size = batch_size
    
  def fit(self, X, Y, verbose):
    """Fit Training Data
    Parameters
    -----------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over training dataset
    random_state : int
      Random number generator to seed for random weight
      
    Attributes
    ------------
    w_ : 1-d array
      weights for each independent input
    b_ : bias variable added to weight output
    cost_ : list
      logarithmic cost value in each epoch
    """
    rgen = np.random.RandomState(self.random_state)
    self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
    self.b_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1) 
    self.cost_ = []
    acc = []
    for iter_ in range(self.n_iter):
      cost_ = 0
      predictions = []
      
      for i in range(int(X.shape[0]/self.batch_size)):
        x = X[i*self.batch_size:(i+1)*self.batch_size]
        y = Y[i*self.batch_size:(i+1)*self.batch_size]
        
        z = self.sigmoid(x)
        pred = [1 if element > 0.5 else 0 for element in z]
        cost_ += self.cost(z, y)
        gradient_, b_gradient = self.gradient(x, y, z)
        self.w_ -= self.lr * np.array(gradient_)   
        self.b_ -= self.lr * np.array(b_gradient)
        predictions.append(pred == y)
        
      acc.append(np.mean(predictions))
      
      if verbose == True and iter_%int(self.n_iter/10) == 0:
        print("Accuracy for epoch ", iter_, " is : ", acc[iter_])
      self.cost_.append(cost_) 
    
  def net_input(self, X):
    return np.dot(X, self.w_) + self.b_
  
  def sigmoid(self, X):
    return 1/(1 + np.exp(-(self.net_input(X))))
  
  def cost(self, z, y):
    return np.sum((-1*(y*np.log(z) + (1 - y)*np.log(1 - z))))/self.batch_size
  
  def gradient(self, x, y, z):
    gradient_ = np.dot(x.T, (z - y))/self.batch_size
    b_grad = np.sum(np.dot(1,(z-y)))/self.batch_size
    return gradient_, b_grad
    
  def predict(self, X):
    """Make predictions for Unknown Data
    
    Parameters
    -----------
    X : Input Data
      array of (N, F), N - Number of samples, F - Number of features for each sample
      
    Attributes
    ------------
    None
    
    Returns
    ------------
    pred : Predictions for Dependent variable
    """
    probabilities = self.sigmoid(X)
    predictions = [1 if element > 0.5 else 0 for element in probabilities]
    return predictions
    pass
