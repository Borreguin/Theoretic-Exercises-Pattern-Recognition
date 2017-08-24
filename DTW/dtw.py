import numpy
from PIL import Image
import sys
import dp_python.dpcore as dpcore
import shutil
import os
import operator

"""
Do dynamic time warping to compare images
"""


class Feature():
    """
    A class that can be constructed with a window
    (that is, a slice of an image)
    one can then use the instance methods go get
    various features for that particular window

    """
    def __init__(self, window):
        self.window = window

    def lower_contour(self):
        """
        distance from bottom of window to lowest black pixel
        normalized to be between 0 and 1
        0: bottom-most pixel is black
        1: all pixels are white
        """
        for (i, value) in enumerate(self.window):
            if not value:
                return float(i) / len(self.window)
        return float(1)

    def upper_contour(self):
        """
        distance from top of window to highest black pixel
        normalized to be between 0 and 1
        0: top-most pixel is black
        1: all pixels are white
        """
        for (i, value) in enumerate(self.window[::-1]):  # ::-1 reverses the list
            if not value:
                return float(i) / len(self.window)
        return float(0)

    def black_white_transitions(self):
        """
        number of transitions between black and white pixels
        normalized to be between 0 and 1
        0: not transitions
        1: all possible transitions
        note if is always changing the number of changes will
        be len(self.window)-1
        """
        last = self.window[0]
        changes = 0
        for (i, value) in enumerate(self.window):
            if value != last:
                changes += 1
            last = value
        return float(changes)/(len(self.window)-1)

    def fraction_black(self):
        """
        fraction of black pixels of all pixels
        normalized to be between 0 and 1
        0: not black present
        1: is black totally
        """
        black_count = sum([1 if x is False else 0 for x in self.window])
        total_count = len(self.window)
        return float(black_count) / total_count

    def fraction_black_between_bounds(self):
        """
        fraction of black pixels of all pixels in between lower_contour and
        upper_contour normalized to be between 0 and 1
        0: not changes
        1: all possible changes
        """
        lc = int(self.lower_contour()*100)
        uc = int(self.upper_contour()*100)
        if (lc >= uc):
            return 0
        bounded_window = self.window[lc:-uc]
        black_count = sum([1 if x is False else 0 for x in bounded_window])
        total_count = len(bounded_window)
        if total_count == 0:
            return 0
        # total_count = len(self.window)
        return float(black_count) / total_count


def get_image_slice(image, index):
    """
    take the image window from image at index pixels
    so calling with index = 0 will take the very first column
    of pixels in an image
    and calling with index = 1 will take the second row
    """
    window_size = 1
    width, height = image.size
    return image.crop((index, 0, index + window_size, height))


def calculate_feature_vector(image):
    """
    take a image,
    calculate a feature object for each slice,
    and return the list of feature objects
l   """
    width, height = image.size
    feature_vector = []
    for index in range(0, width):
        window = get_image_slice(image, index)
        window_array = numpy.array(window)
        feature = Feature(window_array)
        feature_vector.append(feature)
    return feature_vector


def similarity(source, target):
    """
    takes a source window (image slice) and a target window
    and returs the similarity of them
    similarity 0 = identical
    higher = less similar

    TODO: these parameters can certainly be optimized!
    """
    def difference(value):
        return (value(source) - value(target))**2

    lower_contour = difference(lambda x: x.lower_contour())
    upper_contour = difference(lambda x: x.upper_contour())
    black_white_transitions = difference(lambda x: x.black_white_transitions())
    fraction_black = difference(lambda x: x.fraction_black())
    fraction_black_between_bounds = difference(lambda x: x.fraction_black_between_bounds())
    # print lower_contour
    result = 0
    result += upper_contour
    result += lower_contour
    result += black_white_transitions
    # result += float(fraction_black)
    # result += float(fraction_black) / 5
    # result += float(fraction_black_between_bounds)
    # result += float(fraction_black_between_bounds) / 2
    return result**(0.5)


def similarity_matrix(source, target):
    """
    for the two arrays in the arguments, return a matrix of similarity
    result[i, j] is the similarity between source[i] and target[j]
    a similarity is a measure starting at 0
    0 is identical
    higher is less identical
    """
    result = numpy.zeros((len(source), len(target)))
    for i, source_item in enumerate(source):
        for j, target_item in enumerate(target):
            result[i, j] = similarity(source_item, target_item)
    return result


def dynamic_time_warp(similarity_matrix):
    """
    the dynamic programming part of dtw
    using the dpcore library from: https://github.com/dpwe/dp_python
    using the library does the main calculation loop in C
    making it a lot faster!

    # p and q are vectors giving the row and column indices along the best path
    # C returns the full minimum-cost matrix, and phi is the full traceback matrix
    """
    localcost = numpy.array(similarity_matrix, order='C', dtype=float)
    p, q, C, phi = dpcore.dp(localcost, penalty=0.)
    return (p, q)


def warp_image(image, indices):
    """
    take an image and a list of indices
    and create a warped image so that for every item
    in the indices list, that particular window of the
    image noted in the index will be used

    for examnple, if i pass [1, 1, 1, 1, 4] I will get back
    a result  image that consists of the first column of pixels
    from the source image four times, followed by the fourth
    column of pixels from the source image once.
    """
    width, height = image.size
    warped_image = Image.new(image.mode, (len(indices), height))
    for index, value in enumerate(indices):
        image_slice = get_image_slice(image, value)
        warped_image.paste(image_slice, (index, 0))
    return warped_image


def comparison_image(source_image, target_image):
    """
    create a new image with both of the passed images displayed above each other
    usefull vor easiy comparing images visually
    """
    width, height = source_image.size
    result = Image.new(source_image.mode, (width, height * 2))
    result.paste(source_image, (0, 0))
    result.paste(target_image, (0, height))
    return result


def image_similarity(source_image, target_image):
    """
    takes to images, compares them using a sliding window
    (comparing each slice individually with the slice from the other window)
    sum up the similiarities then normalize them according to the image width
    """
    source_vector = calculate_feature_vector(source_image)
    target_vector = calculate_feature_vector(target_image)
    sum = 0
    for index, _ in enumerate(source_vector):
        sum += similarity(source_vector[index], target_vector[index])
    result = float(sum) / len(source_vector)
    return result


def warped_similarity(source, target):
    """
    uses dynamic time warping to make the images as similar as possible
    then compares the resulting warped images
    and returns their similarity
    """
    source_feature_vector = calculate_feature_vector(source)
    target_feature_vector = calculate_feature_vector(target)
    matrix = similarity_matrix(source_feature_vector, target_feature_vector)
    row_indices, column_indices = dynamic_time_warp(matrix)
    warped_source = warp_image(source, row_indices)
    warped_target = warp_image(source, column_indices)
    return image_similarity(warped_source, warped_target)


def comparison():
    """
    runs through all images in the words directory and compares each image to each other image
    displaying the results as csv for easy import to spreadsheet tool
    """
    print("img1, img2, pre-dtw similarity, post-dtw similarity")

    for (source, target) in tertools.combinations(os.listdir("words"), 2):
        source_image = Image.open("words/" + source)
        target_image = Image.open("words/" + target)
        output = str(source)
        output += ", " + str(target)
        output += ", " + str(image_similarity(source_image, target_image))
        output += ", " + str(warped_similarity(source_image, target_image))
        print(output)
        source_image.close()
        target_image.close()


def compare_two_images():
    """
    compare two images and output the result
    """
    source_image = Image.open(sys.argv[2])
    target_image = Image.open(sys.argv[3])
    print("original image similarity:")
    + str(image_similarity(source_image, target_image))
    source_feature_vector = calculate_feature_vector(source_image)
    target_feature_vector = calculate_feature_vector(target_image)
    s_matrix = similarity_matrix(source_feature_vector, target_feature_vector)
    row_indices, column_indices = dynamic_time_warp(s_matrix)
    warped_source = warp_image(source_image, row_indices)
    warped_target = warp_image(target_image, column_indices)
    comparison = comparison_image(warped_source, warped_target)
    print("warped image similarity: ")
    + str(image_similarity(warped_source, warped_target))
    print("warped image length: " + str(len(row_indices)))
    print("comparison image saved to comparison.png")
    comparison.save("comparison.png")


def find_same_images():
    """
    Take an image as a parameter,
    and show images that we suspect that they contain the same word"
    """
    if (os.path.exists("results")):
        shutil.rmtree("results")

    os.makedirs("results")
    source_image = Image.open(sys.argv[2])
    #source_image.save("results/query.png")

    results = {}
    for target in os.listdir("words2"):
        target_image = Image.open("words2/" + target)
        sim = warped_similarity(source_image, target_image)
        results[target] = sim
        target_image.close()

    sorted_results = sorted(results.items(), key=operator.itemgetter(1))
    result_string = sys.argv[2] + "\n"
    for result in sorted_results:
        result_string += str(result[0]) + "," + str(result[1]) + ","

    print result_string
    source_image.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: ")
        print("python dtw.py generate_similarities")
        print("python dtw.py find_same img1.png")
        print("python dtw.py compare_images img1.png img2.png")
    if sys.argv[1] == ("generate_similarities" or "g"):
        comparison()
    elif sys.argv[1] == ("find_same" or "f"):
        find_same_images()
    elif sys.argv[1] == ("compare_images" or "c"):
        compare_two_images()
    else:
        print("invalid argument")
