import numpy as np
from random import uniform
import math


# common utility functions
def distance2(a, b):
    d = np.array(a) - np.array(b)
    return np.dot(d, d)


def distance(a, b):
    return np.sqrt(distance2(a, b))


def constrain(value, lower, upper):
    return max(lower, min(value, upper))


def random_on_circle(rad):
    a = uniform(0, math.pi * 2.0)
    return (rad * math.sin(a), rad * math.cos(a))


def map_value(value, start1, stop1, start2, stop2):
    """
        Simple scaling function
    """
    return start2 + (stop2 - start2) * ((value - start1) / float(stop1 - start1))


def inkey():
    import sys, tty, termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
