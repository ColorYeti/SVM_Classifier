import random
import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10

def ProcessData():
    cifar10_dir="cs231n/datasets/cifar-10-batches-py"
    TrainData, TrainLabel, TestData, TestLabel = load_CIFAR10(cifar10_dir)

    print 'Training Data Shape: ', TrainData.shape
    print 'Training Labels Shape: ', TrainLabel.shape
    print 'Test Data Shape: ', TestData.shape
    print 'Test Labels Shape: ', TestLabel.shape,'\n'

    classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
    NumberClasses=len(classes)
    SamplesPerClass = 7
    '''
    for y,cls in enumerate(classes): # y counts number of classes, cls is the class
        idxs=np.flatnonzero(TrainLabel==y)
        idxs=np.random.choice(idxs,SamplesPerClass,replace=False)
        for i,idx in enumerate(idxs): # i is number of samples, idx is the class
            plt_idx=i*NumberClasses+y+1
            plt.subplot(SamplesPerClass,NumberClasses,plt_idx)
            plt.imshow(TrainData[idx].astype('uint8'))
            plt.axis('off')
            if i==0:
                plt.title(cls)
    plt.show()
    '''
    NumberTraining=49000
    NumberValidation=1000
    NumberTest=1000

    mask = range(NumberTraining,NumberTraining+NumberValidation)
    ValidationData=TrainData[mask]
    ValidationLabel=TrainLabel[mask]

    mask=range(NumberTraining)
    TrainData=TrainData[mask]
    TrainLabel=TrainLabel[mask]

    mask=range(NumberTest)
    TestData=TestData[mask]
    TestLabel=TestLabel[mask]

    print 'Train data shape: ',TrainData.shape
    print 'Train labels shape: ',TrainLabel.shape
    print 'Validation data shape: ',ValidationData.shape
    print 'Validation label shape: ',ValidationLabel.shape
    print 'Test data shape: ',TestData.shape
    print 'Test label shape: ',TestLabel.shape,'\n'

    TrainData=np.reshape(TrainData,(TrainData.shape[0],-1))
    ValidationData=np.reshape(ValidationData,(ValidationData.shape[0],-1))
    TestData=np.reshape(TestData,(TestData.shape[0],-1))

    print 'Train data shape: ',TrainData.shape
    print 'Validation data shape: ',ValidationData.shape
    print 'Test data shape: ',TestData.shape

    MeanImage=np.mean(TrainData,axis=0)
    #print MeanImage[:10],'\n'

    '''
    plt.figure(figsize=(4,4))
    plt.imshow(MeanImage.reshape((32,32,3)).astype('uint8'))
    plt.show(block=False)
    '''

    TrainData-=MeanImage
    ValidationData-=MeanImage
    TestData-=MeanImage

    TrainData=np.hstack([TrainData,np.ones((TrainData.shape[0],1))]).T
    ValidationData=np.hstack([ValidationData,np.ones((ValidationData.shape[0],1))]).T
    TestData=np.hstack([TestData,np.ones((TestData.shape[0],1))]).T

    print TrainData.shape,ValidationData.shape,TestData.shape,'\n'
    SVM(TrainData,TrainLabel,ValidationData,ValidationLabel,TrainData,TrainLabel)


import time
import math
from cs231n.classifiers.linear_svm import svm_loss_naive
from cs231n.gradient_check import grad_check_sparse
from cs231n.classifiers.linear_svm import svm_loss_vectorized

def SVM(Data,Label,VData,VLabel,TData,TLabel):
    W = np.random.randn(10,3073)*0.0001
    loss,grad=svm_loss_naive(W,Data,Label,0.000005)
    print'loss: %f \n'%loss

    '''
    f=lambda w: svm_loss_naive(w, Data,Label,0.0)[0]
    grad_numerical=grad_check_sparse(f,W,grad,10)
    loss, grad = svm_loss_naive(W,Data,Label,5e1)
    f=lambda w:svm_loss_naive(w,Data,Label,5e1)[0]
    grad_numerical=grad_check_sparse(f,W,grad,10)
    '''

    t1=time.time()
    loss_naive,grad_naive=svm_loss_naive(W, Data,Label,0.000005)
    t2=time.time()
    print('\nNaive Loss: %e computed in %fs'%(loss_naive,t2-t1))

    t1=time.time()
    loss_vectorized,grad_vectorized= svm_loss_vectorized(W,Data,Label,0.000005)
    t2=time.time()
    print('Vectorised loss and gradient: %e computed in %fs\n'%(loss_vectorized,t2-t1))

    difference = np.linalg.norm(grad_naive-grad_vectorized,ord='fro')
    print 'difference: %f'%difference

    from cs231n.classifiers import LinearSVM

    svm=LinearSVM()
    t1=time.time()
    loss_hist=svm.train(Data,Label,learning_rate=1e-7,reg=5e4,num_iters=1500,verbose=True)
    t2=time.time()
    print 'That took %fs'%(t2-t1)

    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()

    y_train_predict=svm.predict(Data)
    print 'Training accuracy: %f'%np.mean(Label == y_train_predict)
    y_val_predict=svm.predict(VData)
    print 'Validation accuracy: %f'%np.mean(VLabel == y_val_predict)

    learning_rates=[1e-7,2e-7,5e-7,1e-6]
    regularization_strengths=[1e4,2e4,5e4,1e5,5e5,1e6]

    results={}
    best_val=-1
    best_svm=None

    for learning in learning_rates:
        for regularization in regularization_strengths:
            svm=LinearSVM()
            svm.train(Data,Label,learning_rate=learning,reg=regularization,num_iters=2000)
            y_train_predict=svm.predict(Data)
            train_accuracy=np.mean(y_train_predict==Label)
            print 'Training accuracy: %f'%train_accuracy
            y_val_predict=svm.predict(VData)
            val_accuracy=np.mean(y_val_predict==VLabel)
            print 'Validation accuracy: %f'%val_accuracy

            if val_accuracy>best_val:
                best_val=val_accuracy
                best_svm=svm

            results[(learning,regularization)]=(train_accuracy,val_accuracy)

            for lr,reg in sorted(results):
                train_accuracy,val_accuracy=results[(lr,reg)]
                print 'lr %e reg %e train accuracy: %f val accuracy %f'%(lr,reg,train_accuracy,val_accuracy)
            print 'Best validation accuracy achieved during cross validation: %f '%best_val
            

    x_scatter=[math.log10(x[0]) for x in results]
    y_scatter=[math.log10(x[1]) for x in results]

    sz=[results[x][0]*1500 for x in results]
    plt.subplot(2,1,1)
    plt.scatter(x_scatter,y_scatter,sz)
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('Cifar-10 training accuracy')
    plt.show()

    sz=[results[x][1]*1500for x in results]
    plt.subplot(2,1,1)
    plt.scatter(x_scatter,y_scatter,sz)
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('Cifar-10 validation accuracy')
    plt.show()

    y_test_pred=best_svm.predict(TData)
    test_accuracy=np.mean(y_test_pred==TLabel)
    print 'Linear SVM on raw pixels final test set accuracy: %f'%test_accuracy

    w=best_svm.W[:,:-1]
    w=w.reshape(10,32,32,3)
    w_min,w_max=np.min(w),np.max(w)
    classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
    for i in xrange(10):
        plt.subplot(2,5,i+1)
        wimg = 255.0*(w[i].squeeze()-w_min)/(w_max-w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
    plt.show()

ProcessData()