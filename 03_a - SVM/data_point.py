import matplotlib.pyplot as plot
import numpy
import csv
import pickle
from math import sqrt


class DataPoint:
    def __init__(self, array, value):
        size = int(sqrt(len(array)))
        self.value = value
        self.feature_array = numpy.array(array)
        self.bitmap = [[0 for x in range(size)] for x in range(size)]
        data_item_index = 0
        for row_index in range(size):
            for col_index in range(size):
                self.bitmap[row_index][col_index] = float(
                        self.feature_array[data_item_index])
                data_item_index += 1

    def show(self):
        plot.imshow(self.bitmap)
        plot.title("value: " + self.value)
        plot.gray()
        plot.show()


def load_dataset(filename):
    """
    loads the csv from a passed filename and puts in into a list of DataPoint
    """
    pickle_name = filename + ".pickle"
    try:
        print("trying to load " + filename + " from pickle")
        dataset = pickle.load(open(pickle_name, "rb"))
    except:
        with open(filename, 'r') as csv_file:
            print("no pickle exists. parsing file " + filename)
            dataset = [DataPoint(item[1:], item[0])
                       for item
                       in csv.reader(csv_file, delimiter=',')]
            pickle.dump(dataset, open(pickle_name, "wb"))
    print("loaded " + filename)
    return dataset


def features_density(dataset):
    number_of_features = len(dataset[0][1])
    features = [0.0] * number_of_features

    for training_sample in dataset:
        i = 0
        for feature in training_sample[1]:
            if int(feature) != 0:
                features[i] = features[i] + 1.0
            i += 1

    return features


# Return vector with index of features that are not null, in a specified percentage of samples
def useful_features(dataset, threshold):
    number_of_samples = float(len(dataset))

    features = features_density(dataset)
    useful_feature_index = []

    for k in range(0, len(features)):
        if ((features[k] / number_of_samples) * 100.0) > threshold:
            useful_feature_index.append(k)

    print ("Found ", len(useful_feature_index), " useful features ")
    return useful_feature_index


# Create a new list from a list and index of elements to keep
def sub_list(old_list, index_list):
    new_list = []
    for index in index_list:
        new_list.append(old_list[index])
    return new_list


def fast_load_dataset(file_name):
    samples = []
    with open(file_name, 'r') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',')
        for row in filereader:
            digit = row[0]
            del row[0]
            samples.append((digit, row))
    return samples


def purify_dataset(file_name, file_name_2, threshold):
    print ("Start purify ")

    dataset_1 = fast_load_dataset(file_name)

    useful_feature_index = useful_features(dataset_1, threshold)

    new_file_name_1 = file_name.replace(".csv", "")+"_purified.csv"
    c = csv.writer(open(new_file_name_1, "w"), delimiter=',', quotechar=" ")

    for sample in dataset_1:
        condensed_sample = (sample[0], sub_list(sample[1], useful_feature_index))
        c.writerow([condensed_sample[0], ','.join(condensed_sample[1])])

    dataset_2 = fast_load_dataset(file_name_2)

    new_file_name_2 = file_name_2.replace(".csv", "")+"_purified.csv"
    c = csv.writer(open(new_file_name_2, "w"), delimiter=',', quotechar=" ")
    for sample in dataset_2:
        condensed_sample = (sample[0], sub_list(sample[1], useful_feature_index))
        c.writerow([condensed_sample[0], ','.join(condensed_sample[1])])

    print ("Finish purify")
    return (new_file_name_1, new_file_name_2)
