from __future__ import division
from sklearn import svm
from data_point import DataPoint
from collections import defaultdict
from math import *

import csv
import pickle
import itertools
import numpy
from support_vector_machines import test_kernel

def test_linear_kernel(training_set, test_set, number_of_groups):
    print "\n\nLINEAR KERNEL"
    c_options = [1, 100, 10000, 100000, 1000000]

    kernel_options = [dict(
        C=c_param,
        cache_size=1000,
        decision_function_shape="ovr",
        kernel="linear")
        for c_param
        in c_options]

    def show_kernel_option(kernel_option, accuracy):
        return ("C-param {0} gives accuracy {1}".format(
            kernel_option["C"], accuracy))

    test_kernel(
        kernel_options,
        show_kernel_option,
        training_set,
        test_set,
        number_of_groups)


def test_polynomial_kernel(training_set, test_set, number_of_groups):
    print "\n\nPOLYNOMIAL KERNEL"
    c_options = [0.1, 1, 10]
    degree_options = [1, 2, 4]
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

    test_kernel(
        kernel_options,
        show_kernel_option,
        training_set,
        test_set,
        number_of_groups)


def test_sigmoid_kernel(training_set, test_set, number_of_groups):
    print "\n\nSIGMOID KERNEL"
    c_options = [1, 1000]
    gamma_options = [1, 100, 10000]
    coef0_options = [-1000, -10, 0, 10, 1000]

    param_options = itertools.product(c_options, gamma_options, coef0_options)

    kernel_options = [dict(
        C=c_param,
        gamma=gamma_param,
        coef0=coef0_param,
        cache_size=1000,
        decision_function_shape="ovr",
        kernel="poly")
        for c_param, gamma_param, coef0_param
        in param_options]

    def show_kernel_option(kernel_option, accuracy):
        return ("C-param {0}, gamma param {1} "
                "and coef0 param {2}  give accuracy {3}".format(
                    kernel_option["C"],
                    kernel_option["gamma"],
                    kernel_option["coef0"],
                    accuracy))

    test_kernel(
        kernel_options,
        show_kernel_option,
        training_set,
        test_set,
        number_of_groups)


def test_rbf_kernel(training_set, test_set, number_of_groups):
    print "\n\nRADIAL BASIS FUNCTION KERNEL"
    c_options = [1, 10, 10000]
    gamma_options = [0.1, 10, 1000, 100000]

    param_options = itertools.product(c_options, gamma_options)

    kernel_options = [dict(
        C=c_param,
        gamma=gamma_param,
        cache_size=1000,
        decision_function_shape="ovr",
        kernel="rbf")
        for c_param, gamma_param
        in param_options]

    def show_kernel_option(kernel_option, accuracy):
        return ("C-param {0} and gamma param {1} give accuracy {2}".format(
            kernel_option["C"], kernel_option["gamma"], accuracy))

    test_kernel(
        kernel_options,
        show_kernel_option,
        training_set,
        test_set,
        number_of_groups)


def load_dataset(filename):
    """
    loads the csv from a passed filename and puts in into a list of DataPoint
    """
    pickle_name = filename + ".pickle"
    try:
        print("trying to load " + filename + " from pickle")
        dataset = pickle.load(open(pickle_name, "rb"))
    except:
        with open(filename, 'rb') as csv_file:
            print("no pickle exists. parsing file " + filename)
            dataset = [DataPoint(item[1:], item[0])
                       for item
                       in csv.reader(csv_file, delimiter=',')]
            pickle.dump(dataset, open(pickle_name, "wb"))
    print("loaded " + filename)
    return dataset


def run_test_suite():
    """
    loads the data from the disk and runs all kernel tests
    """
    print "\n\nEXERCISE 2: SUPPORT VECTOR MACHINES\n"
    training_set = load_dataset('1.2.Tr_train.csv')
    test_set = load_dataset('1.2.Tr_test.csv')
    number_of_groups = 5
    print "learning from {0} data points".format(len(training_set))
    print "classifying {0} data points".format(len(test_set))
    print "cross-validating kernel with {0} groups".format(number_of_groups)

    #test_polynomial_kernel(training_set, test_set, number_of_groups)
    #test_rbf_kernel(training_set, test_set, number_of_groups)
    #test_sigmoid_kernel(training_set, test_set, number_of_groups)
    test_linear_kernel(training_set, test_set, number_of_groups)


if __name__ == "__main__":
    run_test_suite()
