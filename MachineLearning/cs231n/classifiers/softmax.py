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
    num_classes = W.shape[0]
    num_train = X.shape[1]

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    ##########################################################################
    for i in xrange(num_train):
        scores = W.dot(X[:, i])
        correct_class_scores = scores[y[i]]
        stability = -scores.max()
        exponential_score = np.exp(scores + stability)
        sum_exponential_score = np.sum(exponential_score, axis=0)

        margin = 0

        for j in xrange(num_classes):
            margin += np.exp(scores[j])
            if (j == y[i]):
                dW[j, :] += (exponential_score[j] /
                             sum_exponential_score) * X.T[i] - X.T[i]
            else:
                dW[j, :] += (exponential_score[j] /
                             sum_exponential_score) * X.T[i]
        loss += -correct_class_scores + np.log(margin)

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += reg * W
    ##########################################################################
    #                          END OF YOUR CODE                                 #
    ##########################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    num_classes = W.shape[0]
    num_train = X.shape[1]
    loss = 0.0
    dW = np.zeros_like(W)

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    ##########################################################################
    scores = W.dot(X)
    stability = -np.max(scores,axis=0)
    correct_class_scores = scores[y, range(num_train)]

    exponential_score = np.exp(scores)
    sum_exponential_score = np.sum(exponential_score)
    probability = exponential_score/sum_exponential_score

    loss = np.log(sum_exponential_score) - correct_class_scores

    dW = probability - 1
    dW = dW.dot(X.T)

    loss /= num_train
    dW /= num_train

    loss += 0.5*reg*np.sum(W*W)
    dW += reg*W
    
    loss = np.sum(loss)
    ##########################################################################
    #                          END OF YOUR CODE                                 #
    ##########################################################################

    return loss, dW
