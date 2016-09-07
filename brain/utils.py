import numpy as np
import cv2
import random
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
    a = random.uniform(0, math.pi * 2.0)
    return (rad * math.sin(a), rad * math.cos(a))


def map_value(value, start1, stop1, start2, stop2):
    """Scale value from a range of start1, stop1, to a range of start2, stop2."""
    return start2 + (stop2 - start2) * ((value - start1) / float(stop1 - start1))


def find_point_closest_to(points, point):
    """Find the a point in points closest to point"""
    best = None
    best_dsq = 0

    for p in points:
        d_x = point[0] - p[0]
        d_y = point[1] - p[1]
        dsq = d_x * d_x + d_y * d_y

        if best is None or dsq < best_dsq:
            best = p
            best_dsq = dsq

    return best


def find_most_distant_pair(points, iter=1000):
    """Find a pair of points in points which are furthest apart.
    This is determined by looking at a couple of random choices, as determined by iter"""

    if len(points) < 2:
        return None
    elif len(points) == 2:
        return (points[0], points[1])

    best = None
    best_dsq = 0

    for i in range(iter):
        a = random.choice(points)
        b = random.choice(points)

        d_x = a[0] - b[0]
        d_y = a[1] - b[1]
        dsq = d_x * d_x + d_y * d_y

        if best is None or dsq > best_dsq:
            best = (a, b)
            best_dsq = dsq

    return best


def find_random_points(points, count=10):
    """Find a number of random points in points"""
    if len(points) < count:
        return points.copy()

    l = points.copy()
    random.shuffle(l)
    return l[:count]


def rotated_subimg(image, center, theta, width, height):
    theta *= math.pi / 180.0  # convert to rad

    v_x = (math.cos(theta), math.sin(theta))
    v_y = (-math.sin(theta), math.cos(theta))
    s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)

    mapping = np.array([[v_x[0], v_y[0], s_x],
                        [v_x[1], v_y[1], s_y]])

    return cv2.warpAffine(image, mapping,(width, height), flags=cv2.WARP_INVERSE_MAP+cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


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
        hash = self._hash(old_point)
        objects = self.contents.get(hash) if hash in self.contents else []

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
        hash = self._hash(point)

        if hash in self.contents:
            return self.contents[hash].copy()
        else:
            return []

    def get_at_with_border(self, point, border_size):
        """Retrieve objects at a certain range defined by border_size around a point in the map."""
        retlist = []

        min_v = self._hash((point[0] - border_size, point[1] - border_size))
        max_v = self._hash((point[0] + border_size, point[1] + border_size))

        for i in range(min_v[0], max_v[0] + 1):
            for j in range(min_v[1], max_v[1] + 1):
                hash = (i, j)

                if hash in self.contents:
                    retlist.extend(self.contents[hash])

        return retlist
