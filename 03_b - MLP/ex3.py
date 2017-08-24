from __future__ import division
from data_point import load_dataset
from classifiers import test_kernel, cross_validate, run_single_test
from sknn.mlp import Classifier, Layer

import itertools


def mlp_kernel_constructor(kernel_option):
    return lambda: Classifier(**kernel_option)


def make_mlp_graph(training_set, test_set, number_of_groups):
    iterations = range(1, 100)

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


def test_mlp(training_set, test_set, number_of_groups):
    print ("\n\nMULTILAYER PERCEPTRON")
    hidden_layer_size_options = [10, 30, 50, 100]
    learning_rate_options = [0.01, 0.1, 0.5]
    training_iteration_options = [10, 40, 100]
    random_state_seeds = [243219, 4353, 342432, 32432, 432]
    param_options = itertools.product(hidden_layer_size_options, learning_rate_options, training_iteration_options, random_state_seeds)

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

    test_kernel(
        mlp_kernel_constructor,
        kernel_options,
        show_kernel,
        training_set,
        test_set,
        number_of_groups)


def run_test_suite():
    """
    loads the data from the disk and runs all kernel tests
    """
    print ("\n\nEXERCISE 3: MULTILAYER PERCEPTRON \n")
    training_set = load_dataset('data/train_short.csv')
    test_set = load_dataset('data/test_short.csv')
    number_of_groups = 10
    print ("learning from {0} data points".format(len(training_set)))
    print ("classifying {0} data points".format(len(test_set)))
    print ("cross-validating kernel with {0} groups".format(number_of_groups))

    #make_mlp_graph(training_set, test_set, number_of_groups)

    test_mlp(training_set, test_set, number_of_groups)


if __name__ == "__main__":
    run_test_suite()
