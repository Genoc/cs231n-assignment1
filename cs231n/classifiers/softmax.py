import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores) # for numeric stability
    correct_score = scores[y[i]]
    softmax_sum = np.sum(np.exp(scores))
    loss += (-correct_score + np.log(softmax_sum) )
    
    dW += np.outer(X[i], np.exp(scores)/softmax_sum)
    dW[:,y[i]] -= X[i]

  loss /= num_train
  dW /= num_train

  # Add regularization
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass

  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  scores -= np.max(scores, axis=1)[:, np.newaxis] # for numeric stability
  correct_scores = scores[np.arange(num_train), y]
  softmax_sums = np.sum(np.exp(scores), axis=1) # 1 for each example
  loss = np.sum(-correct_scores + np.log(softmax_sums))
  
  temp = np.exp(scores)/softmax_sums[:, np.newaxis]
  temp[np.arange(num_train), y] -= 1 
  dW += X.T.dot(temp)

  # normalize
  loss /= num_train
  dW /= num_train

  # Add regularization
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

