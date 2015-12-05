#!/usr/bin/python
import time
import math
import Pyro4
import cv2
import numpy as np
from extract import process_image

from utils import *
from log import log
from stone import Stone, StoneMap

CONTROL_HOSTNAME = 'localhost'
# CONTROL_HOSTNAME = '192.168.0.27'

'''
High-level operations on CNC
'''
class Machine(object):

    def __init__(self, hostname):
        self.uri = 'PYRO:control@{}:5001'.format(hostname)
        self.control = Pyro4.Proxy(self.uri)
        self.cam = Camera(self)
        self.x, self.y, self.z, self.e = 0.0, 0.0, 0.0, 0.0
        self.last_pickup_height = None
        try:
            self.control.home()
        except:
            self.control = None

    def lift_up(self):
        if self.last_pickup_height is not None:
            raise Exception('lift_up called, but previous call not cleared using lift_down')
        self.control.vacuum(True)
        time.sleep(1.0)
        h = self.control.pickup()
        assert h
        self.last_pickup_height = h

    def lift_down(self):
        if self.last_pickup_height is None:
            raise Exception('lift_down called without calling lift_up first')
        self.control.go(z=max(self.last_pickup_height - 3, 0))
        self.control.vacuum(False)
        time.sleep(0.1)
        self.control.pickup_top()
        self.last_pickup_height = None


class Camera(object):

    def __init__(self, machine, index=0):
        self.machine = machine
        self.index = index
        self.resx = 1280.0 # image width (in pixels)
        self.resy = 720.0 # image height (in pixels)
        self.viewx = 128.0 # view width (in cnc units = mm) # FIXME: put real value
        self.viewy = 72.0 # view height (in cnc units = mm) # FIXME: put real value

    # calc distance of perceived pixel from center of the view (in cnc units = mm)
    def pos_to_mm(self, pos, offset=(0, 0)):
        x = self.viewx * (pos[0] / self.resx - 0.5) + offset[0]
        y = self.viewy * (pos[1] / self.resy - 0.5) + offset[1]
        return x, y

    # calc size of perceived pixels (in cnc units = mm)
    def size_to_mm(self, size):
        w = self.viewx * size[0] / self.resx
        h = self.viewy * size[1] / self.resy
        return w, h

    '''
    Compute difference between center of the vacuum head and center of camera view.
    Relative to camera center. In CNC units = milimeters.
    '''
    def vision_delta(self):
        # length of rotating head (in mm)
        length = 40.0 # distance of vacuum tool to rotation axis (Z center)
        # distance between center of Z axis and center of camera view (both in mm)
        dx, dy = -69.23, -1.88
        angle = math.radians(self.machine.e)
        return (dx + length * math.cos(angle) , dy + length * math.sin(angle))

    def grab(self):
        self.machine.control.light(True)
        try:
            cam = cv2.VideoCapture(self.index)
            cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 720)
            cam.set(cv2.cv.CV_CAP_PROP_FPS, 15)
            cam.set(cv2.cv.CV_CAP_PROP_BRIGHTNESS, 39)
            ret, frame = cam.read()
            cam.release()
            if ret:
                ret = frame
        except:
            ret = None
        self.machine.control.light(False)
        return ret

    def grab_extract(self):
        f = self.grab()
        if f is None:
            return []
        s = process_image('grab_{:03d}_{:03d}'.format(self.machine.x, self.machine.y), f)
        return s


class Brain(object):

    def __init__(self):
        self.machine = Machine(CONTROL_HOSTNAME)
        self.map = StoneMap('stonemap.data')
        # shortcuts for convenience
        self.m = self.machine
        self.c = self.machine.control
        # go home (also test if the machine is initialized and working)
        # self.c.home()

    def start(self):
       pass

    def run(self, prgname):
        f = getattr(self, prgname)
        f()

    def scan(self):
        log.debug('Begin scanning')
        self.c.pickup_top()
        self.c.go(e=90)
        self.c.block()
        self.c.feedrate(30000)
        step = 100
        x, y = self.map.size
        stones = []
        for i in range(0, x + 1, step):
            for j in range(0, y + 1, step):
                self.c.go(x=i, y=j)
                self.c.block()
                log.debug('Taking picture at coords {},{}'.format(i, j))
                s = self.machine.cam.grab_extract()
                s['center'] = self.machine.cam.pos_to_mm(s['center'], offset=(i, j))
                s['size'] = self.machine.cam.size_to_mm(s['size'])
                stones.append(s)
                log.debug('Found {} stones'.format(len(stones)))
        log.debug('End scanning')
        # select correct stones
        log.debug('Begin selecting/reducing stones')
        for i in range(len(stones)):
            stones[i]['rank'] = 0.0
        for i in range(len(stones)):
            for j in range(i + 1, len(stones)):
                s = stones[i].similarity(stones[j])
                stones[i]['rank'] += s
                stones[j]['rank'] -= s
        log.debug('End selecting/reducing stones')
        # copy selected stones to storage
        id = 0
        self.map.stones = {}
        for s in stones:
            if s['rank'] > 1.5:
                s1 = Stone(id, s['center'], s['size'], s['angle'], s['color'], s['structure'])
                id += 1
                self.map.add_stone(s1)
        self.map.save()

    def demo1(self):
        # demo program which moves stone back and forth
        while True:
            self._move_stone(0, 0, 0, 500, 250, 90)
            self._move_stone(500, 250, 90, 0, 0, 0)

    def _move_stone(x1, y1, e1, x2, y2, e2):
        self.c.go(e=e1)
        self.c.go(x=x1, y=y1)
        h = self.m.lift_up()
        self.c.go(e=e2)
        self.c.go(x=x2, y=y2)
        self.m.lift_down(h)


if __name__ == '__main__':
    brain = Brain()
    brain.start()
    # brain.run('scan')
