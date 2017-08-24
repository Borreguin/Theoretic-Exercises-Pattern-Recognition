# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 02:06:14 2016

@author: Roberto Sanchez
"""

"""
    simple example of SVM

"""
from sklearn import svm
import Local_Lib as lb
import random as rd
import numpy as np

trMs = []

"""
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
print clf.support_vectors_

#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False)

print "Predict1: " , clf.predict([[2., 2.]])
print "Predict2: " , clf.predict([[1000., 100.]])
print "Predict3: " , clf.predict([[-1., -1.]])
print "Predict3: " , clf.predict([[-400., -4.]])


X2 = [[0], [1], [2], [3]]
Y2 = [0, 1, 2, 3]

#clf2 = svm.SVC(decision_function_shape='ovr')
clf2 = svm.SVC()
clf2.fit(X2, Y2) 
print "supportVectors: \n", clf2.support_vectors_
print "Predict3: " , clf2.predict([2.3])

dec = clf2.decision_function([[1]])
dec.shape[1] # 4 classes: 4*3/2 = 6
print dec
"""

#fileTr = 'tr1.csv'
fileTr = '1.2.Tr_train.csv'
fileTest = '1.2.Tr_test.csv'

#fileTr = 'train.csv'
#fileTest = 'test.csv'

samples = 1000

#Loading the training Set
print "loading the training set"
Tr = lb.openfile(fileTr,samples)
Lbl_Tr = Tr[0]
Ftr_Tr = Tr[1] 
#print Lbl_Tr[0] #to see one example of the train Set
#print Ftr_Tr[0] #to see one example of the train Set

#Initialization of SVM
print "----------Initialization of SVM"
clsf = svm.SVC()
clsf.fit(Ftr_Tr, Lbl_Tr) #trainning

print "---------Loading the test set"

Test = lb.openfile(fileTest,1000) #loading only 10 samples
Lbl_Test = Test[0] #to see one example of the Test Set
Ftr_Test = Test[1] #to see one example of the Test Set

#x = rd.randint(1,len(Lbl_Tt))   #for testing from Test set 
#print Lbl_Tt[x]
#print Ftr_Tt[x]
#print "predict", clsf.predict(Ftr_Tt[x])

print "Accuracy: ", clsf.score(Ftr_Test,Lbl_Test)

print "Automatic parameters: "
aux = str(clsf.get_params())
print np.array(aux)

####################
###change of value of C
clsf.C = 1.5
clsf.gamma = 0.01

print "---new C value:", clsf.C
print "---new gamma value:", clsf.gamma

Train = lb.openfile(fileTr,3000)
Lbl_Train = Tr[0]
Ftr_Train = Tr[1] 

clsf.fit(Ftr_Train, Lbl_Train) #trainning
print "\n \n Accuracy: ", clsf.score(Ftr_Test,Lbl_Test) #testing

####################
###change of value of C

clsf.C = 1.2
clsf.gamma = 0.02
Tr = lb.openfile(fileTr,5000)
Lbl_Train = Tr[0]
Ftr_Train = Tr[1] 

print "---new C value:", clsf.C
print "---new gamma value:", clsf.gamma
clsf.fit(Ftr_Train, Lbl_Train) #trainning
print "\n \n Accuracy: ", clsf.score(Ftr_Test,Lbl_Test) #testing


#######
### Print the last parameters
aux = str(clsf.get_params())
print np.array(aux)



