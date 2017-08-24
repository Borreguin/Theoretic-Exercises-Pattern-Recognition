from __future__ import division
from data_point import load_dataset, load_testset
from classifiers import test_kernel, cross_validate, run_single_test, get_predictions
from sknn.mlp import Classifier, Layer

import itertools
import pickle
import csv
import numpy as np

def mlp_kernel_constructor(kernel_option):
    return lambda: Classifier(**kernel_option)


def make_mlp_graph(training_set, test_set, number_of_groups):
    iterations = [100]

    kernel_options = [dict(
        layers=[
            Layer("Sigmoid", units=784),
            Layer("Sigmoid", units=10),
            Layer("Softmax")],
        learning_rate=0.1,
        n_iter=iteration,
        random_state=483,
        loss_type="mcc",
        # verbose=True # For debugging
        )
        for iteration
        in iterations]

    def show_kernel(kernel_option, accuracy):
        return ""

    print ("iterations, training accuracy, testing accuracy")
    for kernel_option in kernel_options:
        training_accuracy = cross_validate(
                mlp_kernel_constructor(kernel_option),
                training_set,
                number_of_groups)

        classifier = mlp_kernel_constructor(kernel_option)()
        testing_accuracy = run_single_test(classifier, training_set, test_set)
        print ("{0}, {1}, {2}".format(kernel_option['n_iter'], training_accuracy, testing_accuracy))


def kernel_mlp():
    print ("\n\nMULTILAYER PERCEPTRON")
    hidden_layer_size_options = [500]
    learning_rate_options = [0.01]
    training_iteration_options = [100]
    random_state_seeds = [243219]
    param_options = itertools.product(hidden_layer_size_options, 
    	learning_rate_options, training_iteration_options, random_state_seeds)

    kernel_options = [dict(
        layers=[
            Layer("Sigmoid", units=784),
            Layer("Sigmoid", units=hidden_layer_size),
            Layer("Softmax")],
        learning_rate=learning_rate,
        n_iter=training_iterations,
        random_state=random_state_seed,
        loss_type="mcc",
        # verbose=True # For debugging
        )
        for hidden_layer_size, learning_rate, training_iterations, random_state_seed
        in param_options]

    def show_kernel(kernel_option, accuracy):
        return ("{0} neurons in hidden layer, learning rate {1}, {2} iterations and random seed {3} gives accuracy {4}".format(
            kernel_option["layers"][1].units,
            kernel_option["learning_rate"],
            kernel_option["n_iter"],
            kernel_option["random_state"],
            accuracy))

    return kernel_options


def run_test_suite():
    """
    loads the data from the disk and runs all kernel tests
    """
    print ("\n\nEXERCISE 5: HYBRID COMBINATION OF MLP AND SVM \n")
    
    print ("This script takes three trained classifiers \n")
    #prueba = load_dataset('data/train.csv')
    #test_set = load_testset('data/mnist_test.csv')
    test_set = load_dataset('data/stest.csv')  
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

    test_label = [item.value for item in test_set]
    result = zip(labels_1, test_label)
    correct_answers = sum(
            [1 if prediction == value
                else 0
                for prediction, value
                in result])
    accuracy = correct_answers / len(test_label)

    print ("The accuracy classifier_1: {0}".format(accuracy))

    result = zip(labels_2, test_label)
    correct_answers = sum(
            [1 if prediction == value
                else 0
                for prediction, value
                in result])
    accuracy = correct_answers / len(test_label)

    print ("The accuracy classifier_2: {0}".format(accuracy))

    result = zip(labels_3, test_label)
    correct_answers = sum(
            [1 if prediction == value
                else 0
                for prediction, value
                in result])
    accuracy = correct_answers / len(test_label)

    print ("The accuracy classifier_3: {0}".format(accuracy))


    


    print ("\n2. Get predictions using the classifier:")
    #predictions = get_predictions(trainned_classifier,test_set)
    #print predictions
    #filename = "./resultFiles/mnist_test_result.csv"
    #np.savetxt(filename, predictions, delimiter=",",fmt='%s',newline='\n') 



    #make_mlp_graph(training_set, test_set, number_of_groups)

    #test_mlp(training_set, test_set, number_of_groups)


if __name__ == "__main__":
    run_test_suite()
