"""
    it takes the files from .\results_ex3 to make a graph for each file
    
"""
from __future__ import print_function

#import numpy as np
import pylab as pl
import csv
import glob
import six.moves.cPickle as pickle

pref = './results_ex3/*.csv'    

def openfile(file_path):
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        n_iteractions = []
        traning_accuracy = []
        testing_accuracy = []
        [ [n_iteractions.append(float(X[0])), 
           traning_accuracy.append(float(X[1])), 
           testing_accuracy.append(float(X[2]))]  for X in reader ] 
        
        return n_iteractions, traning_accuracy, testing_accuracy
    
def run_graph():
    files = glob.glob(pref)    
    collection = [name.split('/') for name in files]    
    names = [item[-1] for item in collection]   
    n_hidden = ["Hidden Layer 50", "Hidden Layer 500"]
    names = ["Learn_rate 0.01",  "Learn_rate 0.1", "Learn_rate 0.5"]    
    n_iteractions = [2 , 5, 10]
    traning_accuracy = []
    traning_accuracy.append([0.7362, 0.7328, 0.7328]) #0.01
    traning_accuracy.append([0.2632, 0.2688, 0.2688]) #0.1
    traning_accuracy.append([0.116, 0.109, 0.109]) #0.5
    traning_accuracy.append([0.6836, 0.6944, 0.6944]) #0.01
    traning_accuracy.append([0.2948, 0.316, 0.2934]) #0.1
    traning_accuracy.append([0.109, 0.109, 0.109]) #0.5 
    
    j=i=0
    fig = pl.figure()
    for n in n_hidden:                        
        j=j+1        
        ax = fig.add_subplot(1,2,j)
        print(n,j)
        ax.set_title(n)
        ax.set_xlabel("Iteractions")
        ax.set_ylabel("Accurracy")                    
        for name in names:
            ax.plot(n_iteractions , traning_accuracy[i] , 
                    label = name)
            ax.legend(loc='lower left', prop={'size':10})                    
            i = i+1
           
        pl.show()
            
if __name__ == "__main__":
    run_graph()
    


