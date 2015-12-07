import numpy as np

# common utility functions
def distance2(a, b):
    d = np.array(a) - np.array(b)
    return np.dot(d, d)

def distance(a, b):
    return np.sqrt(distance2(a, b))


def constrain(value, lower, upper):
    return max(lower, min(value, upper))


def map_value(value, start1, stop1, start2, stop2):
    """
        Simple scaling function
    """
    return start2 + (stop2 - start2) * ((value - start1) / float(stop1 - start1))
        