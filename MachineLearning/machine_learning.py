import random
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10
from K_Nearest_Neighbour import *
from Linear_Classifier import *
from Neural_Network import *

def process_data(method='NeuralNet'):
    """Process the datasets"""

    cifar10_dir = r"D:\Downloads\cifar-10-python\cifar-10-batches-py"
    train_data, train_label, test_data, test_label = load_CIFAR10(cifar10_dir)

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




process_data()
