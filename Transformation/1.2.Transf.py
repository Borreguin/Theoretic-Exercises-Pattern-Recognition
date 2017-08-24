# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 12:25:01 2016

@author: Roberto Sanchez
This file makes a transformation of the original train file
to get only 28x2 features for each training sample

x(28x28) -> f(x) -> v(28) 
"""

#import linecache
import csv
import numpy as np


#op = 200 #opacity
#fileIn ='Cond_train.csv'
#fileIn ='Met1.Cond_train.csv'
fileIn = 'train.csv'
#fileIn = 'test.csv'

fileOut = "./results/"+ "1.2.Tr_"+fileIn 
#valXY: Given a row, it returns 14 variables such that
#784 pixels in 14 groups  

def extFeature(row,opacy,N):
    #val = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0, 0];       
    
    size = int(len(row)/N) + 1
    #print size    
    u = np.zeros(N) #mean of the vector ""    
    v = np.zeros(N) #variance   
    ui = np.zeros(N)
    ind = []  
    
    val = np.zeros(2*N+1) #this for the answer    
      
    
    for i in range(1,len(row)):  #Sumatory for mean
        #print i        
        if( int(row[i]) > opacy): 
            ix = int(i/size) 
            #print ix
            u[ix] = u[ix] + (i%size) 
            ui[ix] = ui[ix] + 1
            ind.append(i)     #only where there is data       
            
    for i in range(N): #calculation of mean
        if(ui[i]>1):        
            u[i] = round(u[i] / ui[i],1)
    
    for i in ind: # sum variance 
        ix = int(i/size)        
        v[ix] = v[ix] + (i%size - u[ix])**(2) 
    
    for i in range(N): #variance calculation
        if(ui[i]>1):        
            v[i] = round(v[i]**(0.5)/ui[i],3)

    val[0] = row[0]; #keeping label
    
    for i in range(N):        
        ix = i*2 + 1        
        val[ix] = u[i]
        val[ix+1] = u[i]-v[i]
   # print val
    return val
        
with open(fileIn) as csvfile:
    reader = csv.reader(csvfile)
    tr = []    #empty array to save all the evaluations XY
               #for each row in file  
    #i = 0
    for row in reader:
        #i=i+1        
        #if i > 40 and i < 55:
            tr.append(extFeature(row,1,28))        

#print tr
print "Number of transformed samples: " + str(len(tr))
np.savetxt(fileOut, tr, delimiter=",",fmt='%d',newline='\n')    
    
