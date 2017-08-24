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
#import MLPlib as MLP
import numpy as np

from sklearn.cross_validation import train_test_split 
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer

from sknn.mlp import Classifier, Layer

#fileTr = 'tr1.csv'
#fileTrain = '1.2.Tr_train.csv'
fileTrain = '1.2.Tr_Cond_train.csv'
fileTest = '1.2.Tr_test.csv'

#fileTrain = 'train.csv'
#fileTest = 'test.csv'

samples = 2000
print "Open the training set"
[y_Train, X_Train]  = lb.openfile(fileTrain,samples)
print "Open the test set"
[y_Test, X_Test]  = lb.openfile(fileTest,500)

#X_Train -= X_Train.min() # normalize the values to bring them into the range 0-1
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
    


    


