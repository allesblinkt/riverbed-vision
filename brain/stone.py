#!/usr/bin/python

# import pickle as serialization
import serpent as serialization
from random import uniform
from collections import namedtuple
import math
import cv2
import numpy as np

from log import log
from utils import *
from art import art_step

class Stone(object):

    def __init__(self, center, size, angle, color, structure):
        self.center = center
        if size[1] > size[0]:
            size = size[1], size[0]
            angle += 90
        self.size = size
        self.angle = angle % 180
        self.color = color
        self.structure = structure

    def copy(self):
        return Stone(self.center, self.size, self.angle, self.color, self.structure)

    # checks whether stone overlaps with another stone
    def overlaps(self, stone):
        d = distance(self.center, stone.center)
        return d < self.size[0] + stone.size[0] + 2 # add 2 mm

    # computes similarity with another stone
    def similarity(self, stone):
        dc = distance(self.center, stone.center)
        ds = distance(self.size, stone.size)
        da = abs(self.angle - stone.angle)
        if dc > 20:
            return 0.0
        if ds > 20:
            return 0.0
        if da > 20:
            return 0.0
        return 1.0 - max([dc / 20.0, ds / 20.0, da / 20.0])


class StoneMap(object):

    def __init__(self, name):
        self.name = name
        self.stones = []
        self.size = 4000, 2000
        self.workarea = None
        try:
            with open('map/{}.data'.format(self.name), 'rb') as f:
                d = serialization.load(f)
                self.stones = d['stones']
                self.size = d['size']
                self.workarea = d['workarea']
                self.stones = [ Stone(v['center'], v['size'], v['angle'], v['color'], v['structure']) for v in d['stones'] ]
        except:
            self.save()
        for i, s in enumerate(self.stones):
            s.index = i

    # can we put stone to position center?
    def can_put(self, stone):
        for s in self.stones:
            if stone.overlaps(s):
                return False
        return True

    # populate the map with random stones
    def randomize(self, count=2000):
        log.debug('Generating random map of %d stones', count)
        self.stones = []
        failures = 0
        while count > 0 and failures < 100:
            center = (uniform(1000 + 30, self.size[0] - 30), uniform(30, self.size[1] - 30))
            a = uniform(6, 30)
            b = uniform(6, 20)
            if b > a:
                a, b = b, a
            r = uniform(-90, 90)
            c = uniform(0, 255), uniform(42, 226), uniform(20, 223)
            s = [ uniform(0.001, 0.02) for i in range(40) ] + [ uniform(0.15, 0.3), uniform(0.2, 0.4) ]
            s = Stone(center, (a, b), r, c, s)
            good = self.can_put(s)
            if not good:
                failures += 1
            else:
                log.debug('Placing stones: %d left, %d tries', count, failures)
                self.stones.append(s)
                failures = 0
                count -= 1
        log.debug('Have %d stones', len(self.stones))
        self.save()

    def _find_workarea(self):

        # implementation from http://drdobbs.com/database/184410529
        Point = namedtuple('Point', ('X', 'Y'))

        def area(ll, ur):
            if ll.X < 0 or ll.Y < 0 or ur.X < 0 or ur.Y < 0:
                return 0.0
            if ll.X > ur.X or ll.Y > ur.Y:
                return 0.0
            return ((ur.X - ll.X) + 1) * ((ur.Y - ll.Y) + 1)

        def grow_ones(a, c, ll):
            N, M = a.shape
            y = ll.Y - 1
            ur = Point(ll.X - 1, ll.Y - 1)
            x_max = 10000
            while y + 1 < M and a[ll.X, y + 1] == False:
                y += 1
                x = min(ll.X + c[y] - 1, x_max)
                x_max = x
                if area(ll, Point(x, y)) > area(ll, ur):
                    ur = Point(x, y)
            return ur

        def update_cache(a, c, x):
            N, M = a.shape
            for y in range(0, M):
                if a[x][y] == False:
                    c[y] += 1
                else:
                    c[y] = 0

        def mrp_method3(a):
            best_ll = Point(-1, -1)
            best_ur = Point(-1, -1)
            N, M = a.shape
            c = [0] * M
            for llx in range(N-1, -1, -1):
                update_cache(a, c, llx)
                for lly in range(0, M):
                    ll = Point(llx, lly)
                    ur = grow_ones(a, c, ll)
                    if area(ll, ur) > area(best_ll, best_ur):
                        best_ll, best_ur = ll, ur
            return best_ll, best_ur


        # has a bug :-(
        def mrp_method4(a):
            best_ll = Point(-1, -1)
            best_ur = Point(-1, -1)
            N, M = a.shape
            c = [0] * (M + 1)
            stack = []
            for x in range(N-1, -1, -1):
                update_cache(a, c, x)
                width = 0
                for y in range(M + 1):
                    if c[y] > width:
                        stack.append((y, width))
                        width = c[y]
                    if c[y] < width:
                        while True:
                            y0, w0 = stack.pop()
                            if (width * (y - y0)) > area(best_ll, best_ur):
                                best_ll = Point(x, y0)
                                best_ur = Point(x + width - 1, y - 1)
                            width = w0
                            if (c[y] >= width):
                                break
                        width = c[y]
                        if width != 0:
                            stack.append((y0, width))
            return best_ll, best_ur

        log.debug('Finding workarea')
        scale = 10 # work on less precise scale
        usage = np.zeros((int(math.ceil(self.size[0]/scale)), int(math.ceil(self.size[1]/scale))), dtype=np.bool)
        # calculate usage
        for s in self.stones:
            a, b = s.center[0] - s.size[0], s.center[1] - s.size[0]
            c, d = s.center[0] + s.size[0], s.center[1] + s.size[0]
            for x in range(int(math.floor(a/scale)), int(math.floor(c/scale) + 1)):
                for y in range(int(math.floor(b/scale)), int(math.floor(d/scale) + 1)):
                    usage[x][y] = True
        # find workarea
        ll, ur = mrp_method3(usage)
        wa = (ll.X * scale, ll.Y * scale, ur.X * scale, ur.Y * scale)
        return wa

    def image(self, img, scale):
        log.debug('Creating map image')
        for s in self.stones:
            center, size, angle = s.center, s.size, s.angle
            center = int(center[0] / scale), int(center[1] / scale)
            size = int(size[0] / scale), int(size[1] / scale)
            color = 3 * [s.color[0]]
            structure = s.structure
            cv2.ellipse(img, center, size, angle, 0, 360, color, -1)
        if self.workarea:
            a, b, c, d = self.workarea
            a, b, c, d = a / scale, b / scale, c / scale, d / scale
            cv2.rectangle(img, (a,b), (c,d), color=(255, 0, 0))

    def save(self):
        with open('map/{}.data'.format(self.name), 'wb') as f:
            d = {'stones': self.stones, 'size': self.size, 'workarea': self.workarea}
            serialization.dump(d, f)


if __name__ == '__main__':
    map = StoneMap('stonemap_random')
    if len(map.stones) == 0:
        map.randomize()
    while True:
        img_map = np.zeros((map.size[1]/4, map.size[0]/4, 3), np.uint8)
        map.image(img_map, 4)
        cv2.imshow('map', img_map)
        if cv2.waitKey(1) == ord('q'):
            break
        i, nc, na = art_step(map)
        if i is not None:
            if nc is not None:
                map.stones[i].center = nc
            if na is not None:
                map.stones[i].angle = na
