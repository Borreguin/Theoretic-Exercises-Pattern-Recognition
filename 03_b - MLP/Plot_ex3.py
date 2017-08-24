"""
    it takes the files from .\results_ex3 to make a graph for each file
    
"""
from __future__ import print_function

#import numpy as np
import pylab as pl
import csv
import glob

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
         
    for i in range(len(files)):
        n_iteractions, traning_accuracy, testing_accuracy = openfile(files[i])
        fig = pl.figure() 
        ax = fig.add_subplot(1,1,1)
        ax.set_title(names[i])
        ax.set_xlabel("Iteractions")
        ax.set_ylabel("Accuracy")
        ax.plot(n_iteractions , traning_accuracy , 
                label = "traning_accuracy ")
        ax.plot(n_iteractions , testing_accuracy , 
                label = "testing_accuracy " )
        ax.legend(loc='upper right', prop={'size':10})        
            
        pl.show()
        
if __name__ == "__main__":
    run_graph()
    


