import random
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.linear_svm import svm_loss_naive
from cs231n.gradient_check import grad_check_sparse
from cs231n.classifiers.linear_svm import svm_loss_vectorized
from cs231n.classifiers.softmax import softmax_loss_naive
from cs231n.classifiers.softmax import softmax_loss_vectorized


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


