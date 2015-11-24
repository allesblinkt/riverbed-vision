#!/usr/bin/python
import time
import math
import Pyro4

CONTROL_HOSTNAME = 'localhost'

'''
High-level operations on CNC
'''
class Machine(object):

    def __init__(self, hostname):
        self.uri = 'PYRO:control@{}:5001'.format(hostname)
        self.control = Pyro4.Proxy(self.uri)
        self.x, self.y, self.z, self.e = 0.0, 0.0, 0.0, 0.0
        self.last_pickup_height = None
        self.control.home()

    def lift_up(self):
        if self.last_pickup_height is not None:
            raise Exception('lift_up called, but previous call not cleared using lift_down')
        self.control.vacuum(True)
        time.sleep(1.0)
        h = self.control.pickup()
        assert h
        self.last_pickup_height = h

    def lift_down(self.h):
        if self.last_pickup_height is None:
            raise Exception('lift_down called without calling lift_up first')
        time.sleep(0.1)

    '''
    Compute difference between center of the vacuum head and center of camera view.
    Relative to camera center. In CNC units = milimeters.
    '''
    def vision_delta(self):
        # length of rotating head (in mm)
        length = 40.0 # distance of vacuum tool to rotation axis (Z center)
        # distance between center of Z axis and center of camera view (both in mm)
        dx, dy = -50.0, -5.0 # FIXME: put real value
        angle = math.radians(self.e)
        return (dx + length * math.cos(angle) , dy + length * math.sin(angle))


class Camera(object):

    def __init__(self):
        self.resx = 1280.0 # image width (in pixels)
        self.resy = 720.0 # image height (in pixels)
        self.viewx = 128.0 # view width (in cnc units = mm) # FIXME: put real value
        self.viewy = 72.0 # view height (in cnc units = mm) # FIXME: put real value

    # calc distance of perceived pixel from center of the view (in cnc units = mm)
    def pos_to_mm(self, x, y):
        rx = self.viewx * (x / self.resx - 0.5)
        ry = self.viewy * (y / self.resy - 0.5)
        return rx, ry


class Brain(object):

    def __init__(self):
        self.machine = Machine(CONTROL_HOSTNAME)
        # shortcuts for convenience
        self.m = self.machine
        self.c = self.machine.control

    def start(self):
        pass

    def run(self, prgname):
        f = getattr(self, prgname)
        f()

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
    brain.run('demo1')
