import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange
from cs231n.vis_utils import visualize_grid

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None

        #######################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #######################################################################
        p=1

        layer1 = X.dot(W1) + b1
        layer2 = np.maximum(0,layer1)
        #mask = (np.random.rand(*layer2.shape)<p)/p
        #layer2 *= mask
        layer3 = layer2.dot(W2) + b2
        scores = layer3
        #######################################################################
        #                              END OF YOUR CODE                             #
        #######################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #######################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #######################################################################
        exp_correct_score = np.exp(scores[np.arange(N), y])
        exp_scores = np.exp(scores)

        loss = - np.log(exp_correct_score / np.sum(exp_scores, axis=1))
        loss = np.sum(loss)
        loss /= N
        loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        #######################################################################
        #                              END OF YOUR CODE                             #
        #######################################################################

        # Backward pass: compute gradients
        grads = {}
        #######################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #######################################################################
        dscores = (exp_scores.T / np.sum(exp_scores, axis=1)).T
        dscores[np.arange(N), y] -= 1
        dscores /= N
        dscores *= 1.0
        drelu = np.dot(dscores, W2.T) * (layer1 >= 0)

        grads['b2'] = np.sum(dscores, axis=0)
        grads['W2'] = np.dot(layer2.T, dscores) + reg * W2
        grads['b1'] = np.sum(drelu, axis=0)
        grads['W1'] = np.dot(X.T, drelu) + reg * W1
        #######################################################################
        #                              END OF YOUR CODE                             #
        #######################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=True, method='vanilla'):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        v0 = 0
        v1 = 0
        v2 = 0
        v3 = 0
        v0p = v0
        v1p = v1
        v2p = v2
        v3p = v3

        for it in xrange(num_iters):
            it += 1
            X_batch = None
            y_batch = None

            ###################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            ###################################################################
            indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[indices]
            y_batch = y[indices]
            ###################################################################
            #                             END OF YOUR CODE                          #
            ###################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            ###################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            ###################################################################
            if method == 'vanilla':
                self.params['W1'] -= learning_rate * grads['W1']
                self.params['W2'] -= learning_rate * grads['W2']
                self.params['b1'] -= learning_rate * grads['b1']
                self.params['b2'] -= learning_rate * grads['b2']

            elif method == 'momentum':
                mu = 0.5

                v0 = mu * v0 - learning_rate * grads['W1']
                v1 = mu * v1 - learning_rate * grads['W2']
                v2 = mu * v2 - learning_rate * grads['b1']
                v3 = mu * v3 - learning_rate * grads['b2']

                self.params['W1'] += v0
                self.params['W2'] += v1
                self.params['b1'] += v2
                self.params['b2'] += v3

            elif method == 'nesterov':
                mu = 0.5

                v0 = mu * v0 - learning_rate * grads['W1']
                v1 = mu * v1 - learning_rate * grads['W2']
                v2 = mu * v2 - learning_rate * grads['b1']
                v3 = mu * v3 - learning_rate * grads['b2']

                self.params['W1'] += -mu * v0p + (1 + mu) * v0
                self.params['W2'] += -mu * v1p + (1 + mu) * v1
                self.params['b1'] += -mu * v2p + (1 + mu) * v2
                self.params['b2'] += -mu * v3p + (1 + mu) * v3

                v0p = v0
                v1p = v1
                v2p = v2
                v3p = v3

            elif method == 'adagrad':
                v0 += grads['W1']**2
                v1 += grads['W2']**2
                v2 += grads['b1']**2
                v3 += grads['b2']**2

                self.params['W1'] += -learning_rate * \
                    grads['W1'] / (np.sqrt(v0) + 1e-7)
                self.params['W2'] += -learning_rate * \
                    grads['W2'] / (np.sqrt(v1) + 1e-7)
                self.params['b1'] += -learning_rate * \
                    grads['b1'] / (np.sqrt(v2) + 1e-7)
                self.params['b2'] += -learning_rate * \
                    grads['b2'] / (np.sqrt(v3) + 1e-7)
            elif method == 'rmsprop':
                mu = 0.99

                v0 = mu * v0 + (1 - mu) * grads['W1']**2
                v1 = mu * v1 + (1 - mu) * grads['W2']**2
                v2 = mu * v2 + (1 - mu) * grads['b1']**2
                v3 = mu * v3 + (1 - mu) * grads['b2']**2

                self.params['W1'] += -learning_rate * \
                    grads['W1'] / (np.sqrt(v0) + 1e-7)
                self.params['W2'] += -learning_rate * \
                    grads['W2'] / (np.sqrt(v1) + 1e-7)
                self.params['b1'] += -learning_rate * \
                    grads['b1'] / (np.sqrt(v2) + 1e-7)
                self.params['b2'] += -learning_rate * \
                    grads['b2'] / (np.sqrt(v3) + 1e-7)
            elif method == 'adam':
                mu1 = 0.9
                mu2 = 0.995

                v0 = mu1 * v0 + (1 - mu1) * grads['W1']
                v0p = mu2 * v0p + (1 - mu2) * grads['W1']**2
                v0b = v0 / (1 - mu1**it)
                v0pb = v0p / (1 - mu2**it)

                v1 = mu1 * v1 + (1 - mu1) * grads['W2']
                v1p = mu2 * v1p + (1 - mu2) * grads['W2']**2
                v1b = v1 / (1 - mu1**it)
                v1pb = v1p / (1 - mu2**it)

                v2 = mu1 * v2 + (1 - mu1) * grads['b1']
                v2p = mu2 * v2p + (1 - mu2) * grads['b1']**2
                v2b = v2 / (1 - mu1**it)
                v2pb = v2p / (1 - mu2**it)

                v3 = mu1 * v3 + (1 - mu1) * grads['b2']
                v3p = mu2 * v3p + (1 - mu2) * grads['b2']**2
                v3b = v3 / (1 - mu1**it)
                v3pb = v3p / (1 - mu2**it)

                self.params['W1'] += -learning_rate * \
                    v0b / (np.sqrt(v0pb) + 1e-7)
                self.params['W2'] += -learning_rate * \
                    v1b / (np.sqrt(v1pb) + 1e-7)
                self.params['b1'] += -learning_rate * \
                    v2b / (np.sqrt(v2pb) + 1e-7)
                self.params['b2'] += -learning_rate * \
                    v3b / (np.sqrt(v3pb) + 1e-7)
            ###################################################################
            #                             END OF YOUR CODE                          #
            ###################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning
            # rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = np.mean((self.predict(X_batch) == y_batch))
                val_acc = np.mean((self.predict(X_val) == y_val))
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        #######################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        #######################################################################
        layer1 = np.dot(X, self.params['W1']) + self.params['b1']
        layer2 = np.maximum(0,layer1)
        layer3 = np.dot(layer2, self.params['W2']) + self.params['b2']
        y_pred = np.argmax(layer3, axis=1)
        #######################################################################
        #                              END OF YOUR CODE                           #
        #######################################################################

        return y_pred


#
input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5


def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)


def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y


def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
