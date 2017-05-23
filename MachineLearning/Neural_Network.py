import random
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.neural_net import init_toy_data
from cs231n.classifiers.neural_net import init_toy_model
from cs231n.classifiers.neural_net import TwoLayerNet
from cs231n.classifiers.neural_net import rel_error
from cs231n.classifiers.neural_net import show_net_weights
from cs231n.gradient_check import eval_numerical_gradient

def NeuralNet(train_data, train_label, validation_data, validation_label, test_data, test_label):
    input_size = 32 * 32 * 3
    hidden_size = 50
    num_classes = 10
    net = TwoLayerNet(input_size, hidden_size, num_classes)

    print train_label.shape
    # Train the network
    stats = net.train(train_data, train_label, validation_data, validation_label,
                      num_iters=4000, batch_size=1500,
                      learning_rate=5e-3, learning_rate_decay=0.96,
                      reg=2, verbose=True, method='adam')

    # Predict on the validation set
    val_acc = (net.predict(validation_data) == validation_label).mean()
    print('Validation accuracy: ', val_acc)

    # Plot the loss function and train / validation accuracies
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.legend()
    plt.show()

    show_net_weights(net)
#


def TestNeuralNet():
    net = init_toy_model()
    X, y = init_toy_data()

    scores = net.loss(X)
    print('Your scores:')
    print(scores)
    print('correct scores:')
    correct_scores = np.asarray([
        [-0.81233741, -1.27654624, -0.70335995],
        [-0.17129677, -1.18803311, -0.47310444],
        [-0.51590475, -1.01354314, -0.8504215],
        [-0.15419291, -0.48629638, -0.52901952],
        [-0.00618733, -0.12435261, -0.15226949]])
    print(correct_scores)

    # The difference should be very small. We get < 1e-7
    print('Difference between your scores and correct scores:')
    print(np.sum(np.abs(scores - correct_scores)))

    loss, _ = net.loss(X, y, reg=0.05)
    correct_loss = 1.30378789133

    # should be very small, we get < 1e-12
    print('Difference between your loss and correct loss:')
    print(np.sum(np.abs(loss - correct_loss)))
    # Use numeric gradient checking to check your implementation of the backward pass.
    # If your implementation is correct, the difference between the numeric and
    # analytic gradients should be less than 1e-8 for each of W1, W2, b1, and
    # b2.

    loss, grads = net.loss(X, y, reg=0.05)

    # these should all be less than 1e-8 or so
    for param_name in grads:
        def f(W): return net.loss(X, y, reg=0.1)[0]
        param_grad_num = eval_numerical_gradient(
            f, net.params[param_name], verbose=False)
        print('%s max relative error: %e' %
              (param_name, rel_error(param_grad_num, grads[param_name])))

    net = init_toy_model()
    stats = net.train(X, y, X, y,
                      learning_rate=1e-1, reg=5e-6,
                      num_iters=100, verbose=False)

    print('Final training loss: ', stats['loss_history'][-1])

    # plot the loss history
    plt.plot(stats['loss_history'])
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.title('Training Loss history')
    plt.show()