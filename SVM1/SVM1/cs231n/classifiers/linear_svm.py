import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):

    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[0]
    num_train = X.shape[1]
    loss = 0.0
    for i in xrange(num_train):
        scores = W.dot(X[:, i])
        correct_class_score = scores[y[i]]
        bool = (scores - correct_class_score + 1) > 0

        for j in xrange(num_classes):
            if j == y[i]:  # If correct class
                dW[j, :] += -np.sum(np.delete(bool, j)) * X[:, i].T
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[j, :] += bool[j] * X[:, i].T

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):

    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[1]

    loss = W.dot(X) - W.dot(X)[y, np.arange(num_train)] + 1
    bool = loss > 0
    loss = np.sum(loss * bool)
    loss -= 1
    bool = bool * np.ones(loss.shape)
    bool[[y, np.arange(num_train)]] = -(np.sum(bool, axis=0) - 1)
    dW = bool.dot(X.T)

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW
