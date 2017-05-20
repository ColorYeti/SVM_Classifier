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
    '''
    for i in xrange(num_train):
        scoress = W.dot(X[:, i])
        scoress -= scoress.max()
        correct_class_scoress = scoress[y[i]]

        exponential_scores = np.exp(scoress)
        sum_exponential_scores = np.sum(exponential_scores, axis=0)
        print sum_exponential_scores

        margin = 0

        for j in xrange(num_classes):
            margin += np.exp(scoress[j])
            if (j == y[i]):
                dW[j, :] += (exponential_scores[j] /
                             sum_exponential_scores) * X.T[i] - X.T[i]
            else:
                dW[j, :] += (exponential_scores[j] /
                             sum_exponential_scores) * X.T[i]
        loss = -correct_class_scoress + np.log(margin)
        loss += loss

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += reg * W
    '''
    loss = 0.0
    for i in xrange(num_train):
        X_i =  X[:,i]
        score_i = W.dot(X_i)
        stability = -score_i.max()
        exp_score_i = np.exp(score_i+stability)
        exp_score_total_i = np.sum(exp_score_i , axis = 0)
        for j in xrange(num_classes):
            if j == y[i]:
                dW[j,:] += -X_i.T + (exp_score_i[j] / exp_score_total_i) * X_i.T
            else:
                dW[j,:] += (exp_score_i[j] / exp_score_total_i) * X_i.T
        numerator = np.exp(score_i[y[i]]+stability)
        denom = np.sum(np.exp(score_i+stability),axis = 0)
        loss += -np.log(numerator / float(denom))


    loss = loss / float(num_train) + 0.5 * reg * np.sum(W*W)
    dW = dW / float(num_train) + reg * W
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
    scores += - np.max(score, axis=0)
    exponential_scores = np.exp(scores)
    sum_exponential_scores = np.sum(exponential_scores, axis=0)

    loss = np.log(sum_exponential) - score[y,np.arange(num_train)]
    loss = np.sum(loss)

    gradient = exponential_scores / sum_exponential_scores
    gradient[y, np.arange(num_train)] += -1
    dW = gradient.dot(X.T) 
    '''
    score = W.dot(X)
    # On rajoute une constant pr ls overflow
    score += - np.max(score , axis=0)
    exp_score = np.exp(score) # matric exponientiel score
    sum_exp_score_col = np.sum(exp_score , axis = 0) # sum des expo score pr chaque column

    loss = np.log(sum_exp_score_col)
    loss = loss - score[y,np.arange(num_train)]
    loss = np.sum(loss) / float(num_train) + 0.5 * reg * np.sum(W*W)
  
    Grad = exp_score / sum_exp_score_col
    Grad[y,np.arange(num_train)] += -1.0
    dW = Grad.dot(X.T) / float(num_train) + reg*W
    '''
    ##########################################################################
    #                          END OF YOUR CODE                                 #
    ##########################################################################
    loss /= num_train
    dW /= num_train
    
    loss +=  reg * np.sum(W * W)
    dW += reg*W
    
    return loss, dW
