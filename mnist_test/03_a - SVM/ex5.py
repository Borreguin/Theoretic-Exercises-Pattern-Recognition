from __future__ import division
from sklearn import svm
from data_point import load_dataset, load_testset
from classifiers import cross_validate, get_predictions
import itertools
import timeit
import pickle
import csv
import numpy as np

def svc_kernel_constructor(kernel_option):
    return lambda: svm.SVC(**kernel_option)

"""
	According with our experiments we select the following Kernel:
     (Observe that we adapt a bit our framework for this new test)
	kernel = "polynomial"
	c = 1	
    degree = 2
	cache_size = 1000
	decision_function_shape="ovr"
    This script is adapted according to the new data set (i.e. mnist_test.csv)
"""

def polynomial_kernel():
    print ("\n\nPOLYNOMIAL KERNEL")
    c_options = [0.8]
    degree_options = [3]
    param_options = itertools.product(c_options, degree_options)

    kernel_options = [dict(
        C=c_param,
        degree=degree_param,
        cache_size=1000,
        decision_function_shape="ovr",
        kernel="poly")
        for c_param, degree_param
        in param_options]

    def show_kernel_option(kernel_option, accuracy):
        return ("C-param {0} and degree param {1} give accuracy {2}".format(
            kernel_option["C"], kernel_option["degree"], accuracy))
	
    return kernel_options


def run_test_suite():
    """
    loads the data from the disk and runs all kernel tests
    """
    print ("\n\nEXERCISE 5: SUPPORT VECTOR MACHINES\n")

    # (set_1, set_2) = purify_dataset("train_short", "test_short", 50)
    set_1 = "../data/train.csv"
    set_2 = "../data/mnist_test.csv"

    training_set = load_dataset(set_1)
    test_set = load_testset(set_2)

    #set number of groups for the cross-validation
    number_of_groups = 60

    print ("learning from {0} data points".format(len(training_set)))
    print ("classifying {0} data points".format(len(test_set)))
    print ("cross-validating kernel with {0} groups".format(number_of_groups))

    start = timeit.default_timer()

    #trainning the classifier through crossvalidation process 
    print ("\n1. Training the classifier:")

    #creating polynomial kernel for testing 
    kernel_options = polynomial_kernel()
    kernel_option = kernel_options[0]
    """
        cross validate function is defined in "classifiers.py" which is our framework 
        for making the experiments of training and crossvalidation of the Exercise 2a  
    """

    classifier = svc_kernel_constructor(kernel_option) 
    accuracy, trainned_classifier = cross_validate(
            classifier,
            training_set,
            number_of_groups)

    print ("The accuracy after cross-validation process: {0}".format(accuracy))

    print ("Save classifier in pickle:")
    pickle.dump(trainned_classifier, open("svm_trainned_kernel", "wb"))

    trainned_classifier = pickle.load(open("svm_trainned_kernel", "rb"))
    
    print ("\n2. Get predictions using the classifier:")
    predictions = get_predictions(trainned_classifier,test_set)
    print predictions
    filename = "./resultFiles/mnist_test_result.csv"
    np.savetxt(filename, predictions, delimiter=",",fmt='%s',newline='\n') 


    stop = timeit.default_timer()
    print ("test_polynomial_kernel processed in ", stop - start)


if __name__ == "__main__":
    run_test_suite()
