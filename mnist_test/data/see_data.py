#import numpy as np
import pylab as pl
import csv

# This code allow me to see only 9 digits 
opacity = 150
size = 35

with open('mnist_test.csv') as csvfile:
    #reader = csv.DictReader(csvfile)
    reader = csv.reader(csvfile)
    fig = pl.figure()
    n = 0
    for row in reader:
        x= []; y = []   
        print str(row[0])
        #print str(row[1])        
        for i in xrange (len(row) - 1):     
            if(int(row[i]) > opacity and i>0 ):#print row[i+1]
                x.append(i%28) #print i % 28 # including x
                y.append(28-i/28) #print i / 28 #including y         
        if n>=size: break        
        n=n+1        
        #plt = ()*100 + ()*10 + n          
        ax = fig.add_subplot(size/5,size/5,n)
        ax.plot(x, y , 'o', label = str(row[1]),color = 'r')
        pl.show()
        #print valXY(row,200)
        

