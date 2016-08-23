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
    """Clamps a value between lower and upper."""
    return max(lower, min(value, upper))


def random_on_circle(rad):
    """Generate a random angle 2D vector (tuple) with a specified length rad."""
    a = uniform(0, math.pi * 2.0)
    return (rad * math.sin(a), rad * math.cos(a))


def map_value(value, start1, stop1, start2, stop2):
    """Scale value from a range of start1, stop1, to a range of start2, stop2."""
    return start2 + (stop2 - start2) * ((value - start1) / float(stop1 - start1))


def inkey():
    import sys
    import tty
    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


class SpatialHashMap(object):
    """SpatialHashMap"""

    def __init__(self, cell_size=10):
        super(SpatialHashMap, self).__init__()

        self.cell_size = cell_size
        self.contents = {}

    def _hash(self, pointtuple):
        return int(pointtuple[0] / self.cell_size), int(pointtuple[1] / self.cell_size)

    def update_object_at_point(self, old_point, new_point, obj):
        objects = self.contents.setdefault(self._hash((old_point.x, old_point.y)), [])

        if obj in objects:
            objects.remove(obj)

        self.insert_object_at_point(new_point, obj)

    def insert_object_at_point(self, point, obj):
        """Insert an object at a certain point into the map. Allows double insertion."""
        hash = self._hash(point)

        if hash not in self.contents:
            self.contents[hash] = []

        if obj not in self.contents[hash]:  # TODO: consider set...
            self.contents[hash].append(obj)

    def remove(self, obj):
        """Remove an object obj from the hash. This happens by a linear search, so it's not very performant."""

        for hash, objects in self.contents.items():
            if obj in objects:
                objects.remove(obj)

            if len(objects) == 0:
                del self.contents[hash]

    def get_at(self, point):
        """Retrieve objects that at a certain point in the hash"""
        hash = self._hash((point.x, point.y))

        if hash in self.contents:
            return self.contents[hash].copy()
        else:
            return []

    def get_at_with_border(self, point, border_size):
        """Retrieve objects at a certain range defined by border_size around a point in the map."""
        retlist = []

        min = self._hash((point.x - border_size, point.y - border_size))
        max = self._hash((point.x + border_size, point.y + border_size))

        for i in range(min[0], max[0] + 1):
            for j in range(min[1], max[1] + 1):
                hash = (i, j)

                if hash in self.contents:
                    retlist.append(self.contents[hash])

        return retlist
