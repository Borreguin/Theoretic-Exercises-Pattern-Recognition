# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 07:59:36 2016

@author: Roberto Sanchez

Utility to see the result from Ex3.py
"""

import pylab as pl
import six.moves.cPickle as pickle

n_epochs=30
filetoOpen = 'res_'+ "ep" + str(n_epochs) + ".pkl"
n_iter,validation_acc,test_acc = pickle.load(open(filetoOpen))

fig = pl.figure()

ax = fig.add_subplot(1,1,1)
ax.plot(n_iter, validation_acc , '-', label = "Training_Error",color = 'r')
ax.plot(n_iter, test_acc , '-', label = "Testing_Error",color = 'g')
ax.set_title(filetoOpen)
ax.set_xlabel("Iteractions")
ax.set_ylabel("Error")
        
ax.legend()
pl.show()
        


