from __future__ import division

# import itertools
import numpy


def graph_results():
    return 0


def test_kernel(kernel_constructor, kernel_options, show_kernel_option,
                training_set, test_set, number_of_groups):
    """
    create a kernel for each kernel_option and test it on the datasets
    :param kernel_options: a dictionary of constructor params for the kernel
    :param show_kernel_options: a function that pretty prints kernel options
    :param training_set: list of DataPoint to train from
    :param test_set: list of DataPoint to test the kernel on
    :param number_of_groups: the number of groups for cross-validation
    """

    print ("\ncross-validating...")

    best_result = 0, None

    for kernel_option in kernel_options:
        accuracy = cross_validate(
            kernel_constructor(kernel_option),
            training_set,
            number_of_groups)

        print (show_kernel_option(kernel_option, accuracy))

        if accuracy >= best_result[0]:
            best_result = accuracy, kernel_option

    best_accuracy, best_kernel_option = best_result

    print ("\nbest result:")
    print (show_kernel_option(best_kernel_option, best_accuracy))
    classifier = kernel_constructor(best_kernel_option)()
    accuracy = run_single_test(classifier, training_set, test_set)
    print ("best parameters applied to test "
           "set give accuracy of {0}".format(accuracy))


def train_classifier(classifier, datapoints):
    """
    takes a classifier and trains it on the passed data points
    """
    training_data = numpy.array([item.feature_array for item in datapoints])
    class_labels = numpy.array([item.value for item in datapoints])
    classifier.fit(training_data, class_labels)


def test_classifier(classifier, datapoints):
    """
    takes a classifier, predicts values of the passed datapoints
    and then compares the predictions with their real values
    returns the accuracy of the predictions as a fraction
    """
    test_data = numpy.array([item.feature_array for item in datapoints])
    predictions = classifier.predict(test_data)
    result = zip(predictions, [item.value for item in datapoints])
    correct_answers = sum(
            [1 if prediction == value
                else 0
                for prediction, value
                in result])
    accuracy = correct_answers / len(predictions)
    return accuracy


def run_single_test(classifier, training_set, test_set):
    """
    trains the classifier on the training set
    then returns its accuracy on the test set
    """
    train_classifier(classifier, training_set)
    accuracy = test_classifier(classifier, test_set)
    return accuracy


def cross_validate(create_classifier, training_set, number_of_groups):
    """
    takes a functions that can be called to create a classifier (as multiple
    classifiers will need to be created) and a number of groups. It divides
    the training set into that many subgroups and does cross-validation on them
    returns the average accuracy for predicting all groups
    """
    accuracy_sum = 0
    for iteration in range(number_of_groups):
        testing_subset = training_set[iteration::number_of_groups]
        training_subset = [item
                          for item
                          in training_set
                          if item not in testing_subset]
        classifier = create_classifier()
        accuracy = run_single_test(classifier, training_subset, testing_subset)
        accuracy_sum += accuracy
    return accuracy_sum / number_of_groups
