## Exercise 3_a: Support Vector Machines

With this exercise we want to build the foundation for the Pattern Recognition Framework. To do this we should still work on the MNIST dataset, with which 
we should be familiar by now. In this exercise we should aim to improve the recognition rate on the MNIST dataset using SVM.

The code to do testing and cross-validation is in classifiers.py. The code that actually tests the various kernels is in ex2.py. 


## Exercise 3_b: Multi Layer Perceptron

With this exercise we use our framework for applying an MLP approach to the MNIST dataset. The goal of this exercise is to train an MLP with one hidden layer. And experiment with different parameters.

We experimented with implementing MLP manually according to a tutorial, and also tried to use the MLP implementation in theano. The results of this can be found in the folder "MLP with theano".

We then decided to use the scikit-neuralnetwork package that is not in scikit itself but uses a different library for MLP. However, since it has a compatible interface to scikit, we were able to use the same framework for testing and cross-validation etc. as we used in exercise 1. The MLP testing with the different parameters is in ex3.py.

We also created a function to plot error rates on test and validation set after n iterations. The script to plot is in Plot_ex3.py, and a plot can be seen under results_ex3/overfitting.png. However, it was not possible to run the script for enough iterations to actually notice overfitting due to time constraints. Note that even like this, the plot describes a scenario with only 100 learning datapoints and 10 testing datapoints.


