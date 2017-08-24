# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 09:37:55 2016

@author: Roberto Sanchez 
Local Library with useful functions

"""
import csv
import numpy as np
import random
#import linecache
import datetime

trMs = []

def openfile(fileName,samples): 
    Ms = []
    print "opening..."
    with open(fileName) as csvfile:
        reader = csv.reader(csvfile)
        S=[] #features
        L = [] #labels
        sizeX = 0
        if samples <= -1 :
            for row in reader: 
                sizeX = len(row)
                S.append(list(map(float,row[1:sizeX])))
                L.append(int(row[0]))
                
        if samples >= 1:
            n = 0
            for row in reader :
                sizeX = len(row)
                S.append(list(map(float,row[1:sizeX]))) 
                L.append(int(row[0]))
                n = n + 1                
                if n >= samples:
                    break
                        
                                
        aux = "Document size: " + str(len(S))
        
        Ms.append(aux)    
        aux = "Size of each sample: " + str(len(S[0])) 
        
        Ms.append(aux)
        trMs.append(Ms)
        #print Ms        
        return [np.array(L), np.array(S)]  #return the vector with the content inside


def saveResults(prefix,fileOut,toSave):    
    trN = [] 
    trN.append( "Current Time: " + str(datetime.datetime.now()) )   
    trN.append(prefix)
    #print trMs
    print trN
    print prefix    
    np.savetxt(prefix + "Log.txt", trMs, delimiter=",",fmt='%s',newline='\n')       
    np.savetxt(prefix + fileOut, toSave, delimiter=",",fmt='%s',newline='\n')   
    #np.savetxt(prefix + "aux.txt", auxiliar, delimiter=",",fmt='%s',newline='\n')   
    #np.savetxt(prefix + "larg.txt", larg, delimiter=",",fmt='%s',newline='\n')   
        
    return "Save..."















