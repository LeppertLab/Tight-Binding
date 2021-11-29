import numpy as np


def show_error_statistics(difference):
    print("\nError statistics")
    print("Mean: {}".format(np.mean(difference)))
    print("Median: {}".format(np.median(difference)))
    print("Standard deviation: {}".format(np.std(difference)))
    print("Of absolute")
    abs_difference = np.abs(difference)
    print("Mean: {}".format(np.mean(abs_difference)))
    print("Median: {}".format(np.median(abs_difference)))
    print("Standard deviation: {}".format(np.std(abs_difference)))
