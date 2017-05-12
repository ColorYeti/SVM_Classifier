import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10

def process_data(method = 'SVM'):
    """Process the datasets"""
    cifar10_dir = (r"D:\School projects\Programming\Python"+
                   r"\Machine-Learning-Stuff\SVM1\SVM1\cs231n\datasets\cifar-10-batches-py")
    train_data, train_label, test_data, test_label = load_CIFAR10(cifar10_dir)

    print 'Training Data Shape: ', train_data.shape
    print 'Training Labels Shape: ', train_label.shape
    print 'Test Data Shape: ', test_data.shape
    print 'Test Labels Shape: ', test_label.shape, '\n'

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
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
    
    if (method == 'SVM'):
        number_training = 49000
        number_validation = 1000
        number_test = 1000

        mask = range(number_training, number_training+number_validation)
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
        validation_data = np.reshape(validation_data, (validation_data.shape[0], -1))
        test_data = np.reshape(test_data, (test_data.shape[0], -1))

        print 'Train data shape: ', train_data.shape
        print 'Validation data shape: ', validation_data.shape
        print 'Test data shape: ', test_data.shape

        mean_image = np.mean(train_data, axis=0)
        print mean_image[:10], '\n'

        plt.figure(figsize=(4, 4))
        plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))
        plt.show()

        train_data -= mean_image
        validation_data -= mean_image
        test_data -= mean_image

        train_data = np.hstack([train_data, np.ones((train_data.shape[0], 1))]).T
        validation_data = np.hstack([validation_data, np.ones((validation_data.shape[0], 1))]).T
        test_data = np.hstack([test_data, np.ones((test_data.shape[0], 1))]).T

        print train_data.shape, validation_data.shape, test_data.shape, '\n'
        SVM(train_data, train_label, validation_data, validation_label, train_data, train_label)
    elif (method == 'KNN'):
        KNN(train_data, train_label, test_data, test_label)
    elif (method == 'Softmax'):
        number_training = 49000
        number_validation = 1000
        number_test = 1000
        number_development = 500

        mask = list(range(number_training, number_training+number_validation))
        validation_data = train_data[mask]
        validation_label = train_label[mask]

        mask = list(range(number_training))
        train_data = train_data[mask]
        train_label = train_label[mask]

        mask = list(range(number_test))
        test_data = test_data[mask]
        test_label = test_label[mask]

        mask = np.random.choice(number_training, number_development, replace=False)
        development_data = train_data[mask]
        development_label = train_label[mask]

        train_data = np.reshape(train_data, (train_data.shape[0], -1))
        validation_data = np.reshape(validation_data, (validation_data.shape[0], -1))
        test_data = np.reshape(test_data, (test_data.shape[0], -1))
        development_data = np.reshape(development_data, (development_data.shape[0], -1))        

        mean_image = np.mean(train_data, axis=0)
        train_data -= mean_image
        validation_data -= mean_image
        test_data -= mean_image
        development_data -= mean_image

        train_data = np.hstack([train_data, np.ones((train_data.shape[0], 1))])
        validation_data = np.hstack([validation_data, np.ones((validation_data.shape[0], 1))])
        test_data = np.hstack([test_data, np.ones((test_data.shape[0], 1))])
        development_data = np.hstack([development_data, np.ones((development_data.shape[0], 1))])

        print 'Train data shape: ', train_data.shape
        print 'Train labels shape: ', train_label.shape
        print 'Validation data shape: ', validation_data.shape
        print 'Validation label shape: ', validation_label.shape
        print 'Test data shape: ', test_data.shape
        print 'Test label shape: ', test_label.shape, '\n'

        Softmax(train_data, train_label, validation_data, validation_label, test_data, test_label, development_data, development_label)

from cs231n.classifiers.linear_svm import svm_loss_naive
from cs231n.gradient_check import grad_check_sparse
from cs231n.classifiers.linear_svm import svm_loss_vectorized

def SVM(train_data, train_label, validation_data, validation_label, test_data, test_label):
    W = np.random.randn(10,3073)*0.0001
    loss, grad = svm_loss_naive(W, train_data, train_label, 0.000005)
    print 'loss: %f \n'%loss
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
    loss_hist = svm.train(train_data, train_label, learning_rate=1e-7, reg=5e4, num_iters=1500, verbose=True)
    t2 = time.time()
    print 'That took %fs'%(t2-t1)

    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()

    y_train_predict = svm.predict(train_data)
    print 'Training accuracy: %f'%np.mean(train_label == y_train_predict)
    y_val_predict = svm.predict(validation_data)
    print 'Validation accuracy: %f'%np.mean(validation_label == y_val_predict)

    learning_rates = [1e-7, 2e-7, 5e-7, 1e-6]
    regularization_strengths = [1e4, 2e4, 5e4, 1e5, 5e5, 1e6]

    results = {}
    best_val = -1
    best_svm = None

    for learning in learning_rates:
        for regularization in regularization_strengths:
            svm = LinearSVM()
            svm.train(train_data, train_label, learning_rate=learning, reg=regularization, num_iters=2000)
            y_train_predict = svm.predict(train_data)
            train_accuracy = np.mean(y_train_predict == train_label)
            print 'Training accuracy: %f'%train_accuracy
            y_val_predict = svm.predict(validation_data)
            val_accuracy = np.mean(y_val_predict == validation_label)
            print 'Validation accuracy: %f'%val_accuracy

            if val_accuracy > best_val:
                best_val = val_accuracy
                best_svm = svm

            results[(learning, regularization)] = (train_accuracy, val_accuracy)

    for lr,reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print 'lr %e reg %e train accuracy: %f val accuracy %f'%(lr, reg, train_accuracy, val_accuracy)
    print 'Best validation accuracy achieved during cross validation: %f '%best_val
            

    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]

    sz=[results[x][0]*1500 for x in results]
    plt.subplot(1, 1, 1)
    plt.scatter(x_scatter, y_scatter, sz)
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('Cifar-10 training accuracy')
    plt.show()

    sz=[results[x][1]*1500 for x in results]
    plt.subplot(1, 1, 1)
    plt.scatter(x_scatter, y_scatter, sz)
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('Cifar-10 validation accuracy')
    plt.show()

    y_test_pred = best_svm.predict(test_data)
    test_accuracy = np.mean(y_test_pred == test_label)
    print 'Linear SVM on raw pixels final test set accuracy: %f'%test_accuracy

    w = best_svm.W[:, :-1]
    w = w.reshape(10, 32, 32, 3)
    w_min, w_max = np.min(w), np.max(w)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in xrange(10):
        plt.subplot(2,5,i+1)
        wimg = 255.0*(w[i].squeeze()-w_min)/(w_max-w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
    plt.show()
from cs231n.classifiers import KNearestNeighbor
from cs231n.gradient_check import grad_check_sparse

def KNN(train_data, train_label, test_data, test_label):
    pass

from cs231n.classifiers.softmax import softmax_loss_naive
from cs231n.classifiers.softmax import softmax_loss_vectorized
from cs231n.classifiers.softmax import softmax_loss_vectorized

def Softmax(train_data, train_label, validation_data, validation_label, test_data, test_label, development_data, development_label):
    W = np.random.randn(10,3073)*0.0001
    loss, grad = softmax_loss_naive(W, development_data.T, development_label, 0.000005)
    print 'loss: %f \n'%loss
    print 'sanity check: %f'%(-np.log(0.1))

    f = lambda w: softmax_loss_naive(w, development_data.T, development_label, 0.0)[0]
    grad_numerical = grad_check_sparse(f, W, grad, 10)

    loss, grad = softmax_loss_naive(W, development_data.T, development_label, 5e1)
    f = lambda w: softmax_loss_naive(w, development_data.T, development_label, 5e1)[0]
    grad_numerical = grad_check_sparse(f, W, grad, 10)

    tic = time.time()
    loss_naive, grad_naive = softmax_loss_naive(W, development_data.T, development_label, 0.000005)
    toc = time.time()
    print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))

    tic = time.time()
    loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, development_data.T, development_label, 0.000005)
    toc = time.time()
    print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

    grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
    print('Loss difference: %f' % np.abs(loss_naive - loss_vectorized))
    print('Gradient difference: %f' % grad_difference)

process_data()