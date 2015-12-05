import numpy as np

# common utility functions
def distance2(a, b):
    d = np.array(a) - np.array(b)
    return np.dot(d, d)

def distance(a, b):
    return np.sqrt(distance2(a, b))
