from __future__ import division
from data_point import load_dataset, load_testset
from classifiers import test_kernel, cross_validate, run_single_test, get_predictions
from sknn.mlp import Classifier, Layer

import itertools
import pickle
import csv
import numpy as np


def run_test_suite():
    """
    loads the data from the disk and runs all kernel tests
    """
    print ("\n\nEXERCISE 5: HYBRID COMBINATION OF MLP AND SVM \n")
    
    print ("This script takes three trained classifiers \n")
    
    test_set = load_testset('../data/mnist_test.csv')
    #test_set = load_dataset('data/stest.csv')  
    print ("classifying {0} data points".format(len(test_set)))

    print ("\n1. Loading the classifier:")
    # Loading the trained classifiers
    classifier_1 = pickle.load(open("svm_trainned_classifier_1.pickle", "rb"))
    classifier_2 = pickle.load(open("svm_trainned_classifier_2.pickle", "rb"))
    classifier_3 = pickle.load(open("mlp_trainned_classifier.pickle", "rb"))
   
    # get prediction the predicted labels  
    labels_1 = get_predictions(classifier_1,test_set)
    labels_2 = get_predictions(classifier_2,test_set)
    labels_3 = get_predictions(classifier_3,test_set)

    # label3 has different format therefore we transfor the labels
    labels_3 = [x[0] for x in labels_3]
    labels_3 = np.array(labels_3)

    print ("\n2. Get predictions using the classifiers:")

    test_label = [item.value for item in test_set]

    result = zip(labels_1, labels_2, labels_3)
    predictions = []
    for y1, y2, y3 in result:
        #because the classifier 1 is the most accurate
        value = y1 
        if (y1 == y2 and y2 == y3):
            value = y1
        if (y1 == y2):
            value = y1
        if (y1 == y3):
            value = y1
        if (y2 == y3):
            value = y2
        
        predictions.append(value)

    filename = "./resultFiles/mnist_test_result.csv"
    print ("\nSave results in ", filename)
    np.savetxt(filename, predictions, delimiter=",",fmt='%s',newline='\n') 

if __name__ == "__main__":
    run_test_suite()
