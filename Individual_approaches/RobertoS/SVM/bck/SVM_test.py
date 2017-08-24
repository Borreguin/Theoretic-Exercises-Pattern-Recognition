# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 04:12:37 2016

@author: Roberto Sanchez 
Testing the actual behavior
"""
from sklearn import svm
import Local_Lib as lc
import random as rd
import numpy as np

#fileTr = '1.2.Tr_Cond_train.csv'
#fileTr = '1.2.Tr_train.csv'
#fileTt = '1.2.Tr_test.csv'

fileTr = 'train.csv'
fileTt = 'test.csv'

prefix = "./results/" + fileTr[:5] + "_"

samples = 3000

#Loading the training Set
print "loading the training set"
Tr = lc.openfile(fileTr,samples)
Lbl_Tr = Tr[0] #the trainning samples
Ftr_Tr = Tr[1] #the labels

#print "train:", Lbl_Tr[0] #to see one example of the train Set
#print Ftr_Tr[0] #to see one example of the train Set

#Initialization of SVM
print "Initialization of SVM"
clsf = svm.SVC()
clsf.fit(Ftr_Tr, Lbl_Tr) 

print "loading the test set"
Tt = lc.openfile(fileTt,1500) #loading only 10 samples
Lbl_Tt = Tt[0] #the Test Set
Ftr_Tt = Tt[1] #the labels Test Set
#print "test:", Lbl_Tt[0] #to see one example of the train Set
#print Ftr_Tt[0] #to see one example of the train Set

#x = rd.randint(1,100)   #for testing from Test set 
#print Lbl_Tt[x]
#print Ftr_Tt[x]
#print "predict", clsf.predict(Ftr_Tt[x])


#n = 0
#for i in range(len(Lbl_Tt)):
#    label =  clsf.predict(Ftr_Tt[i])   
    #print "i:",i, Lbl_Tt[i] ,"_", label
#    if (str(label[0]) == Lbl_Tt[i]):
#        n  = n+1;
        
#print n        
#print "Precision: ", n/float(len(Lbl_Tt))        
#Score allow us to see the precision:

print "Precision: ", clsf.score(Ftr_Tt,Lbl_Tt)

aux = str(clsf.get_params())
print np.array(aux)
#lc.saveResults(prefix,"parameters.txt",aux)

clsf.C = 0.5
print "Precision: ", clsf.score(Ftr_Tt,Lbl_Tt)


