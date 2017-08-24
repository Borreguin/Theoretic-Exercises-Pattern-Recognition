# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 02:06:14 2016

@author: Roberto Sanchez
"""

"""
    simple example of MLP

"""
#from sklearn.neural_network import MLPClassifier
import Local_Lib as lb
import MLPlib as MLP
import numpy as np

from sklearn.cross_validation import train_test_split 
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer

def XOR_NN():
    Neu_Net = MLP.NeuralNetwork([2,2,1], 'tanh')
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])
    Neu_Net.fit(X, Y)
    
    for i in [[0, 0], [0, 1], [1, 0], [1,1]]:
        print(i,Neu_Net.predict(i))

def Sk_example():
    digits = load_digits()
    X = digits.data
    y = digits.target
    X -= X.min() # normalize the values to bring them into the range 0-1
    X /= X.max()
    
    print len(X)
    print len(y)
    
    
    # number of features  
    # number of hidden neurons
    # Number of outputs
    
    nn = MLP.NeuralNetwork([64,100,10],'tanh')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)
    
    nn.fit(X_train,labels_train,epochs=30000) #training the neural network
    
    #make the testing
    predictions = []
    for i in range(X_test.shape[0]):
        o = nn.predict(X_test[i] )
        predictions.append(np.argmax(o))
    print confusion_matrix(y_test,predictions)
    print "\n"
    print classification_report(y_test,predictions)

def Sk_exampl2():
    digits = load_digits()
    X = digits.data
    y = digits.target
    X -= X.min() # normalize the values to bring them into the range 0-1
    X /= X.max()
    
    print len(X)
    print len(y)
    
    
    # number of features  
    # number of hidden neurons
    # Number of outputs
    
    nn = MLP.NeuralNetwork([64,200,10],'tanh')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)
 
    print X_train[800]
    print y_train[800]
    print X_train[500]
    print y_train[500]
   
    print "len train:", len(X_train)
    nn.fit(X_train,labels_train,epochs=30000) #training the neural network
    
    #make the testing
    predictions = []
    for i in range(X_test.shape[0]):
        o = nn.predict(X_test[i] )
        #p = np.argmax(o)
        #print p
        predictions.append(np.argmax(o))
    print confusion_matrix(y_test,predictions)
    print "\n"
    print classification_report(y_test,predictions)



trMs = []


#fileTr = 'tr1.csv'
#fileTrain = '1.2.Tr_train.csv'
#fileTrain = '1.2.Tr_Cond_train.csv'
#fileTest = '1.2.Tr_test.csv'

fileTrain = 'train.csv'
fileTest = 'test.csv'

samples = 2000
print "Open the training set"
[y_Train, X_Train]  = lb.openfile(fileTrain,samples)
print "Open the test set"
[y_Test, X_Test]  = lb.openfile(fileTest,500)

X_Train -= X_Train.min() # normalize the values to bring them into the range 0-1
X_Train /= X_Train.max()
    

#labels_Train = LabelBinarizer().fit_transform(y_Train)
#labels_Test = LabelBinarizer().fit_transform(y_Test)

Xsize = len(X_Train[0])
Ysize = 10 #for the ten digits
N_hidden_neurons = 900

print "Number of features:", Xsize
print "Trained set:", len(y_Train)

#print X_Train[800]
#print y_Train[800]
#print X_Train[500]
#print y_Train[500]
    
NeuN = MLP.NeuralNetwork([Xsize,N_hidden_neurons,Ysize],'tanh')
NeuN.fit(X_Train,y_Train,learning_rate=0.5,epochs=30000) #training the neural network
    
#make the testing
predictions = []
for i in range(X_Test.shape[0]):
    o = NeuN.predict(X_Test[i] )
    #p = np.argmax(o)
    #print p
    predictions.append(np.argmax(o))
print confusion_matrix(y_Test,predictions)
print "\n"
print classification_report(y_Test,predictions)


    
#Sk_exampl2()




