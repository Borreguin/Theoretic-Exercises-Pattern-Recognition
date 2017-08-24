# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:04:51 2016

@author: Roberto Sanchez 
This exercise is using the tutorial of Multilayer Perceptron 
from http://deeplearning.net/tutorial/mlp.html

In this script, we make a separation between the initialization of the MLP 
Classifier and the initialization of the methods to evaluate the MLP classifier
 
"""
from __future__ import print_function

__docformat__ = 'restructedtext en'


import os
import sys
import timeit
import six.moves.cPickle as pickle

import MLPLib as MLP
import Local_Lib as lb
import theano
import theano.tensor as T
import numpy as np

"""   
   HERE THE SYMBOLICALY DEFINITION OF COST
   The cost function defined as theano.tensor.var.TensorVariable
     
"""
def init_cost(classifier, y, L1_reg, L2_reg):
    """       
    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    """
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4
    return cost    

     
def init_model(index, set_x, set_y, 
                    batch_size, outputs, x, y):

    """
    This model allow to make the following actions: testing, training and 
    validating:
        index: Is the set of indexes to test  
        setx: the set of features according with each index 
        sety: The set of goals (labels)
        output: Is the result of applying the model         
        x: a symbolical variable for x         
        y: a symbolical varibble for y
        
    COMPILE THE THE THEANO FUNCTION    
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    """
    model = theano.function(
        inputs=[index],
        outputs=outputs,
        givens={
            x: set_x[index * batch_size:(index + 1) * batch_size],
            y: set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    return model  
    
    
def MLP_classifier(n_input, n_output, n_hidden_neurons = 500):
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    """
        This a constructor of the MLP Classifier 
        This classifier uses the definition of MLPLib. library    
    
        A random number generator for initializating the weights 
        is important, this definition is inside of the constructor of the 
        MLP classifier.
    """
   
    print('... building the model')
      
    x = T.matrix('x')
    
    # construct the MLP class
    classifier = MLP.MLP(
        input=x,
        n_in= n_input,
        n_hidden=n_hidden_neurons, #default
        n_out=n_output
    )
    
    return classifier        
        

def test_mlp( MLP_classifier, DataSet,
             learning_rate=0.01,
              n_epochs=5, batch_size=20, 
              L1_reg=0.00, L2_reg=0.0001,
              verbose_validation=1000):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron
    
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

   """
    "Opening the Data set"
    "each set contains: x_features (list of arrays), y_goals (list)"
    train_set, test_set, valid_set = DataSet
      
    "Opening the MLP Classifier"
    classifier = MLP_classifier    
    
    """
        HERE DECLARATION OF THE TENSOR VARIABLES
    allocate symbolic variables for the data, the symbolic variables
    are going to use in the theano model:
        # Features (X) as the data is presented as rasterized images(Matrix)
        # Output (Y) as the labels are presented as 10 vector of [int] labels
        # index to a [mini]batch
    """
    index = T.lscalar() #<TensorType(int64, scalar)>
    x = classifier.input  
    y = T.ivector('y')    
    
    "Sharing Theano variables"    
    train_set_x, train_set_y = MLP.shared_datasetTheano(train_set)
    test_set_x, test_set_y = MLP.shared_datasetTheano(test_set)
    valid_set_x, valid_set_y = MLP.shared_datasetTheano(valid_set)
  
        
    #print( train_set_x)
    
    # compute number of minibatches for training, validation and testing
    """
        Beacuse we are using shared memory, we can access to the shared memory
        so we avoid a copy of the shared memory.
        if the parameter of borrow = false, we have a deep copy of the shared 
        variable:
        n_train_batches: is the size of one batch
        n_valid_batches: is the size of one batch
        n_test_batches: is the size of one batch
        
    """
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
  
    print("Train batch zise: ", n_train_batches)    
    print("Test batch zise: ",  n_test_batches)    
#    print("Valid batch zise: ", n_valid_batches)    
    
   
    
# start-snippet-5
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs    
    
    "Initialization of the cost function"
    cost = init_cost(classifier, y, L1_reg, L2_reg)
         
    
    gparams = [T.grad(cost, param) for param in classifier.params]
    
    """
    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    """    
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    
    "Compile the test model, using a theano function"
    test_model = init_model(index, test_set_x, 
                                 test_set_y, batch_size,
                                 classifier.errors(y), x, y)
    "Compile the valid model"                            
    validate_model = init_model(index, valid_set_x, 
                                 valid_set_y, batch_size,
                                 classifier.errors(y), x, y)
    "Compile the train model"    
    train_model = init_model(index, valid_set_x, 
                                 valid_set_y, batch_size,
                                 cost , x, y)
    
    
    """
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    """    
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    " # end-snippet-5"



    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    verbose = verbose_validation  # look as this many examples regardless
    verbose_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.9995  # a relative improvement of this much is
                                   # considered significant
    verbose_frequency = min(n_train_batches, verbose // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    
    n_iter = []
    validation_loss = []
    test_loss = []
    print("Number of batches to train: ", n_train_batches)
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #minibatch = [1,2,3...n_train_batches]
        for minibatch_index in range(n_train_batches):
            
            if(minibatch_index % verbose_frequency == 0 ):
                print("Training over minibatch: %i" %(minibatch_index))            
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

#####################################################################
####### VERBOSE ACTION TO SEE THE IMPROVE OF THE MLP CLASSIFIER
##################################
            if (iter) % verbose_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                val_score = 1 - this_validation_loss
               # print("Validiting over the validate_set")
                print(
                    'epoch: %i, Val_batch: %i/%i, accuracy: %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        (val_score)* 100.
                    )
                )

#####################################################################
####### TESTINTG OVER THE TEST DATA SET
##################################

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        verbose = max(verbose, iter * verbose_increase)

                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                test_losses = [test_model(i) for i
                               in range(n_test_batches)]
                test_score = 1 - np.mean(test_losses)
                #print("Testing over the testing data_set")                
                print(('     epoch %i, Test_batch %i/%i, '
                       'accuracy %f %%') %
                      (epoch, minibatch_index + 1, n_train_batches,
                       test_score * 100.))
                #with open('best_model.pkl', 'wb') as f:
                #    pickle.dump(classifier, f)
                        

                #######   SAVE RESULTS: #######
                """ Here we save the values to plot the graph                 
                """
                n_iter.append(iter)
                validation_loss.append(this_validation_loss)
                test_loss.append(1-test_score)
                
                #############################    
           # if verbose <= iter:
           #     done_looping = True
           #     break



    end_time = timeit.default_timer()
    print(('OPTIMIZATION COMPLETE.\n Best validation score of %f %% '
           'obtained at iteration %i, \n with acurracy of: %f %%') %
          ((1-best_validation_loss) * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file = sys.stderr)
    result = [n_iter,validation_loss,test_loss]
    print("SAVE RESULTS:")  
    filetoSave = 'res_'+ "ep" + str(n_epochs) + ".pkl"      
    with open(filetoSave, 'wb') as f:
        pickle.dump(result, f)
                        
    
    

if __name__ == '__main__':
    fileTrain = '../../data/train.csv'
    fileTest = '../../data/test.csv'
    
    samples = 10000
    print ("Open the training set")
    train_set  = lb.openfile(fileTrain,samples,init=0)
    print ("Open the valid set")
    valid_set  = lb.openfile(fileTrain,2000,init=10000)    
    print ("Open the test set")
    test_set  = lb.openfile(fileTest,samples,init=0)
    
    learning_rate=0.01
    n_hidden = 500; n_epochs=30 ; 
    
    """
    batch_size=20 #this parameter is fixed (number of samples 
                  to treath in each iteraction)  
    L1_reg=0.00; L2_reg=0.0001 # This parameter is by default
    
    """
    
    inputsize = 28*28
    outputsize = 10
    #Good parameters: 
    #   Small batch_size
    #   learning rate = 0.01    
    #   medium number of epochs (number of times that over train the trainnig set)
    
    MLPclassifier = MLP_classifier(inputsize,outputsize,n_hidden)
    DataSet = [train_set, test_set, valid_set]
    test_mlp(MLPclassifier, DataSet,
             learning_rate, n_epochs,
             verbose_validation = 500)








