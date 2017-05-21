import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers.linear_svm import svm_loss_naive
from cs231n.gradient_check import grad_check_sparse
from cs231n.classifiers.linear_svm import svm_loss_vectorized
from cs231n.classifiers import KNearestNeighbor
from cs231n.classifiers.softmax import softmax_loss_naive
from cs231n.classifiers.softmax import softmax_loss_vectorized
from cs231n.classifiers.neural_net import init_toy_data
from cs231n.classifiers.neural_net import init_toy_model
from cs231n.classifiers.neural_net import TwoLayerNet
from cs231n.classifiers.neural_net import rel_error
from cs231n.classifiers.neural_net import show_net_weights
from cs231n.gradient_check import eval_numerical_gradient

np.set_printoptions(threshold=np.inf)

def process_data(method='NeuralNet'):
    """Process the datasets"""

    cifar10_dir = r"D:\Downloads\cifar-10-python\cifar-10-batches-py"
    train_data, train_label, test_data, test_label = load_CIFAR10(cifar10_dir)

    print train_label

    print 'Training Data Shape: ', train_data.shape
    print 'Training Labels Shape: ', train_label.shape
    print 'Test Data Shape: ', test_data.shape
    print 'Test Labels Shape: ', test_label.shape, '\n'

    classes = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    number_classes = len(classes)
    sample_per_class = 7
    """
    for y, cls in enumerate(classes): # y counts number of classes, cls is the class
        idxs = np.flatnonzero(train_label == y)
        idxs = np.random.choice(idxs, sample_per_class, replace=False)
        for i, idx in enumerate(idxs): # i is number of samples, idx is the class
            plt_idx = i*number_classes+y+1
            plt.subplot(sample_per_class, number_classes, plt_idx)
            plt.imshow(train_data[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()
    """
    #
    if method == 'KNN':
        KNN(train_data, train_label, test_data, test_label)
    #
    else:
        number_training = 49000
        number_validation = 1000
        number_test = 1000

        mask = range(number_training, number_training + number_validation)
        validation_data = train_data[mask]
        validation_label = train_label[mask]

        mask = range(number_training)
        train_data = train_data[mask]
        train_label = train_label[mask]

        mask = range(number_test)
        test_data = test_data[mask]
        test_label = test_label[mask]

        print 'Train data shape: ', train_data.shape
        print 'Train labels shape: ', train_label.shape
        print 'Validation data shape: ', validation_data.shape
        print 'Validation label shape: ', validation_label.shape
        print 'Test data shape: ', test_data.shape
        print 'Test label shape: ', test_label.shape, '\n'

        train_data = np.reshape(train_data, (train_data.shape[0], -1))
        validation_data = np.reshape(
            validation_data, (validation_data.shape[0], -1))
        test_data = np.reshape(test_data, (test_data.shape[0], -1))

        print 'Train data shape: ', train_data.shape
        print 'Validation data shape: ', validation_data.shape
        print 'Test data shape: ', test_data.shape

        mean_image = np.mean(train_data, axis=0)
        print mean_image[:10], '\n'

        plt.figure(figsize=(4, 4))
        plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))
        # plt.show()

        train_data -= mean_image
        validation_data -= mean_image
        test_data -= mean_image

        train_data = train_data.T
        validation_data = validation_data.T
        test_data = test_data.T

        print train_data.shape, validation_data.shape, test_data.shape, '\n'

        if (method == 'SVM'):
            SVM(train_data, train_label, validation_data,
                validation_label, train_data, train_label)
        #
        elif (method == 'Softmax'):
            Softmax(train_data, train_label, validation_data,
                    validation_label, test_data, test_label)

        else:
            NeuralNet(train_data.T, train_label, validation_data.T,
                      validation_label, test_data.T, test_label)

#


def SVM(train_data, train_label, validation_data, validation_label, test_data, test_label):
    W = np.random.randn(10, 3072) * 0.0001
    loss, grad = svm_loss_naive(W, train_data, train_label, 0.000005)
    print 'loss: %f \n' % loss
    '''
    f=lambda w: svm_loss_naive(w, train_data,train_label,0.0)[0]
    grad_numerical=grad_check_sparse(f,W,grad,10)
    loss, grad = svm_loss_naive(W,train_data,train_label,5e1)
    f=lambda w:svm_loss_naive(w,train_data,train_label,5e1)[0]
    grad_numerical=grad_check_sparse(f,W,grad,10)

    t1 = time.time()
    loss_naive, grad_naive = svm_loss_naive(W, train_data, train_label, 0.000005)
    t2 = time.time()
    print '\nNaive Loss: %e computed in %fs'%(loss_naive, t2-t1)

    t1 = time.time()
    loss_vectorized,grad_vectorized = svm_loss_vectorized(W, train_data, train_label, 0.000005)
    t2 = time.time()
    print 'Vectorised loss and gradient: %e computed in %fs\n'%(loss_vectorized, t2-t1)

    difference = np.linalg.norm(grad_naive-grad_vectorized, ord='fro')
    print 'difference: %f'%difference
    '''
    from cs231n.classifiers import LinearSVM

    svm = LinearSVM()
    t1 = time.time()
    loss_hist = svm.train(train_data, train_label,
                          learning_rate=1e-7, reg=5e4, num_iters=1000, verbose=True)
    t2 = time.time()
    print 'That took %fs' % (t2 - t1)

    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()

    train_label_predict = svm.predict(train_data)
    print 'Training accuracy: %f' % np.mean(train_label == train_label_predict)
    validation_label_predict = svm.predict(validation_data)
    print 'Validation accuracy: %f' % np.mean(validation_label == validation_label_predict)

    learning_rates = [1e-7, 2e-7, 5e-7, 1e-6]
    regularization_strengths = [1e4, 2e4, 5e4, 1e5, 5e5, 1e6]

    results = {}
    best_val = -1
    best_svm = None

    for learning in learning_rates:
        for regularization in regularization_strengths:
            svm = LinearSVM()
            svm.train(train_data, train_label, learning_rate=learning,
                      reg=regularization, num_iters=2000)
            train_label_predict = svm.predict(train_data)
            train_accuracy = np.mean(train_label_predict == train_label)
            print 'Training accuracy: %f' % train_accuracy
            validation_label_predict = svm.predict(validation_data)
            val_accuracy = np.mean(validation_label_predict == validation_label)
            print 'Validation accuracy: %f' % val_accuracy

            if val_accuracy > best_val:
                best_val = val_accuracy
                best_svm = svm

            results[(learning, regularization)] = (
                train_accuracy, val_accuracy)

    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print 'lr %e reg %e train accuracy: %f val accuracy %f' % (lr, reg, train_accuracy, val_accuracy)
    print 'Best validation accuracy achieved during cross validation: %f ' % best_val

    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]

    sz = [results[x][0] * 1500 for x in results]
    plt.subplot(1, 1, 1)
    plt.scatter(x_scatter, y_scatter, sz)
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('Cifar-10 training accuracy')
    plt.show()

    sz = [results[x][1] * 1500 for x in results]
    plt.subplot(1, 1, 1)
    plt.scatter(x_scatter, y_scatter, sz)
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('Cifar-10 validation accuracy')
    plt.show()

    y_test_pred = best_svm.predict(test_data)
    test_accuracy = np.mean(y_test_pred == test_label)
    print 'Linear SVM on raw pixels final test set accuracy: %f' % test_accuracy

    print best_svm.W.shape
    w = best_svm.W[:, :]
    print w.shape
    w = w.reshape(10, 32, 32, 3)
    w_min, w_max = np.min(w), np.max(w)
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    for i in xrange(10):
        plt.subplot(2, 5, i + 1)
        wimg = 255.0 * (w[i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
    plt.show()


#

def KNN(train_data, train_label, test_data, test_label):
    pass


#

def Softmax(train_data, train_label, validation_data, validation_label, test_data, test_label):
    W = np.random.randn(10, 3072) * 0.0001
    '''
    loss, grad = softmax_loss_naive(W, train_data, train_label, 0.000005)
    print 'loss: %f \n' % loss
    print 'sanity check: %f' % (-np.log(0.1))

    def f(w): return softmax_loss_naive(w, train_data, train_label, 0.0)[0]
    grad_numerical = grad_check_sparse(f, W, grad, 10)

    loss, grad = softmax_loss_naive(W, train_data, train_label, 5e1)

    def f(w): return softmax_loss_naive(w, train_data, train_label, 5e1)[0]
    grad_numerical = grad_check_sparse(f, W, grad, 10)
    '''
    tic = time.time()
    loss_naive, grad_naive = softmax_loss_naive(
        W, train_data, train_label, 0.000005)
    toc = time.time()
    print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))

    tic = time.time()
    loss_vectorized, grad_vectorized = softmax_loss_vectorized(
        W, train_data, train_label, 0.000005)
    toc = time.time()
    print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

    grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
    print('Loss difference: %f' % np.abs(loss_naive - loss_vectorized))
    print('Gradient difference: %f' % grad_difference)


#
def NeuralNet(train_data, train_label, validation_data, validation_label, test_data, test_label):
    input_size = 32 * 32 * 3
    hidden_size = 50
    num_classes = 10
    net = TwoLayerNet(input_size, hidden_size, num_classes)

    print train_label.shape
    # Train the network
    stats = net.train(train_data, train_label, validation_data, validation_label,
                      num_iters=2500, batch_size=200,
                      learning_rate=1e-4, learning_rate_decay=0.95,
                      reg=0.25, verbose=True, method='adam')

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
    plt.show()

    show_net_weights(net)

    for learning in learning_rates:
        for regularization in regularization_strengths:
            svm = LinearSVM()
            svm.train(train_data, train_label, learning_rate=learning,
                      reg=regularization, num_iters=2000)
            train_label_predict = svm.predict(train_data)
            train_accuracy = np.mean(train_label_predict == train_label)
            print 'Training accuracy: %f' % train_accuracy
            validation_label_predict = svm.predict(validation_data)
            val_accuracy = np.mean(validation_label_predict == validation_label)
            print 'Validation accuracy: %f' % val_accuracy

            if val_accuracy > best_val:
                best_val = val_accuracy
                best_svm = svm

            results[(learning, regularization)] = (
                train_accuracy, val_accuracy)

    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print 'lr %e reg %e train accuracy: %f val accuracy %f' % (lr, reg, train_accuracy, val_accuracy)
    print 'Best validation accuracy achieved during cross validation: %f ' % best_val
#


def NN():
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
#


#

process_data()
