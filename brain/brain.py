#!/usr/bin/python
import time
import Pyro4

CONTROL_HOSTNAME = 'localhost'

class Machine(object):

    def __init__(self, hostname):
        self.uri = 'PYRO:control@{}:5001'.format(hostname)
        self.control = Pyro4.Proxy(self.uri)
        self.x, self.y, self.z, self.e = 0.0, 0.0, 0.0, 0.0
        self.last_pickup_height = None
        self.control.home()

    def lift_up():
        self.control.vacuum(True)
        time.sleep(1.0)
        h = self.control.pickup()
        assert h
        self.last_pickup_height = h

    def lift_down(h):
        if self.last_pickup_height is not None:
            self.control.go(z=max(self.last_pickup_height - 3, 0))
            self.control.vacuum(False)
            time.sleep(0.1)
            self.control.pickup_top()
            self.last_pickup_height = None
        raise Exception('lift_down called without calling lift_up first')


class Brain(object):

    def __init__(self):
        self.machine = Machine(CONTROL_HOSTNAME)
        # shortcuts for convenience
        self.m = self.machine
        self.c = self.machine.control

    def start(self):
        pass

    def run(self):
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
    brain.run()
