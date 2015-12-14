#!/usr/bin/python

import pickle as serialization
# import serpent as serialization
from random import uniform
from collections import namedtuple
import math
import cv2
import numpy as np
import time
import shutil
#from rtree import index

from log import log
from utils import *
from art import art_step

class Stone(object):
    def __init__(self, center, size, angle, color, structure, flag=False):
        self.center = center
        if size[1] > size[0]:
            angle += 90
            self.size = size[1], size[0]
        else:
            self.size = size
        self.angle = angle % 180
        self.color = color
        self.structure = structure
        self.bogus = False
        self.flag = flag

    def copy(self):
        return Stone(self.center, self.size, self.angle, self.color, self.structure, self.flag)

    # checks whether stone overlaps with another stone
    def overlaps(self, stone):
        dx = self.center[0] - stone.center[0]
        dy = self.center[1] - stone.center[1]
        dsq = dx * dx + dy * dy
        rs = (self.size[0] + stone.size[0] + 2)
        # d = distance(self.center, stone.center)
        return dsq < rs * rs # add 2 mm

    #
    def coincides(self, stone):
        d = distance(self.center, stone.center)
        return d < self.size[1] + stone.size[1]

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

    def save(self, filename):
        with open(filename, 'wb') as f:
            serialization.dump(self, f)


class StoneHole(object):

    def __init__(self, stone):
        self.center = stone.center
        self.size = min(stone.size)


class StoneMap(object):



    def __init__(self, name):
        self.name = name
        self.stones = []
        self.holes = []
        self.size = 3770, 1730
        self.stage = 0

        print "Loading"

        # self.idx = index.Index()

        try:
            meta = False
            with open('map/{}.data'.format(self.name), 'rb') as f:
                d = serialization.load(f)
                self.size = d['size']

                if not isinstance(d['stage'], (int, long, float, complex)):
                    self.stage = d['stage']
                else:
                    self.stage = None

                if d.has_key('stones'):
                    log.debug('Loading stones from OLD format')
                    self.stones = d['stones']
                else:
                    log.debug('Loading stones from NEW format #1')
                    self.stones = d['stones1']
                    meta = True
            if meta:
                with open('map/{}.data2'.format(self.name), 'rb') as f:
                    d = serialization.load(f)
                    log.debug('Loading stones from NEW format #2')
                    sm = d['stones2']
                    assert len(self.stones) == len(sm)
                    for i in range(len(sm)):
                        self.stones[i].color = sm[i]['color']
                        self.stones[i].structure = sm[i]['structure']
        except Exception, e:
            log.warn('Something happened while loading')
            log.warn(e)
            self.save(meta=True)

        for i in range(len(self.stones)):
            self.update_idx(i)

        log.debug('Loaded %d stones', len(self.stones))
        self._metadata()


    # precompute useful info, but don't store it
    def _metadata(self):
        self.maxstonesize = 0
        for i, s in enumerate(self.stones):
            s.index = i

            if not hasattr(s, 'flag'):
                s.flag = False

            if s.size[0] > self.maxstonesize:
                self.maxstonesize = s.size[0]
        self.maxstonesize *= 2.0
        self.maxstonesize = 38.0 * 2.0  # Manual, so we can resume with different scan


    def update_idx(self, i):
        s = self.stones[i]
        # self.idx.insert(i, (s.center[0], s.center[1], s.center[0], s.center[1]))

    # Can we put a stone, compare against the given list of stones
    def can_put_list(self, stone, stones):
        if stone.center[0] - stone.size[0] <= 0:
            return False
        if stone.center[1] - stone.size[0] <= 0:
            return False
        if stone.center[0] + stone.size[0] >= self.size[0]:
            return False
        if stone.center[1] + stone.size[0] >= self.size[1]:
            return False

        sr = 50
        for s in stones:
            if stone.overlaps(s):
                return False
        return True

    # can we put stone to position center?
    def can_put(self, stone):
        return can_put_list(stone, self.stones)

    # populate the map with random stones
    def randomize(self, count=2000):
        log.debug('Generating random map of %d stones', count)
        self.stones = []
        failures = 0
        while count > 0 and failures < 100:
            center = (uniform(30, self.size[0] - 1000 - 30), uniform(30, self.size[1] - 30))
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
        self.save(meta=True)
        self._metadata()

    def image(self, img, scale):
        log.debug('Creating map image')
        for s in self.stones:
            center, size, angle = s.center, s.size, s.angle
            center = int((self.size[0] - center[0]) / scale), int(center[1] / scale)
            size = int(size[0] / scale), int(size[1] / scale)
            dummy = np.array([np.array([s.color], dtype=np.uint8)])
            color = (cv2.cvtColor(dummy, cv2.COLOR_LAB2BGR)[0, 0]).tolist()
            structure = s.structure
            cv2.ellipse(img, center, size, 360 - angle, 0, 360, color, -1)
        for h in self.holes:
            center = int((self.size[0] - h.center[0]) / scale), int(h.center[1] / scale)
            size = int(h.size / scale)
            cv2.circle(img, center, size, (255, 255, 255))
        """
        if self.workarea:
            a, b, c, d = self.workarea
            a, b, c, d = a / scale, b / scale, c / scale, d / scale
            cv2.rectangle(img, (a,b), (c,d), color=(255, 0, 0))
        """

    def save(self, meta=False):
        with open('map/{}.data'.format(self.name), 'wb') as f:
            s = [ Stone(x.center, x.size, x.angle, None, None, x.flag) for x in self.stones ]
            d = {'stones1': s, 'size': self.size, 'stage': self.stage}
            serialization.dump(d, f)
        ts = int(time.time())
        # backup with timestamp
        shutil.copy('map/{}.data'.format(self.name), 'map/{}-{}.data'.format(self.name, ts))
        if meta:
            with open('map/{}.data2'.format(self.name), 'wb') as f:
                s = [ {'color': x.color, 'structure': x.structure} for x in self.stones ]
                d = {'stones2': s}
                serialization.dump(d, f)
            # backup with timestamp
            shutil.copy('map/{}.data2'.format(self.name), 'map/{}-{}.data'.format(self.name, ts))

if __name__ == '__main__':
    map = StoneMap('stonemap')
    if len(map.stones) == 0:
        # map.randomize()
        print "No STONES!"
    while True:
        img_map = np.zeros((map.size[1]/2, map.size[0]/2, 3), np.uint8)
        map.image(img_map, 2)
        cv2.imshow('map', img_map)
        if cv2.waitKey(1) == ord('q'):
            break
        i, nc, na = art_step(map)
        if i is not None:
            map.holes.append(StoneHole(map.stones[i]))
            if nc is not None:
                map.stones[i].center = nc
                map.update_idx(i)
            if na is not None:
                map.stones[i].angle = na
    map.save()
