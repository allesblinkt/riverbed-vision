#!/usr/bin/python
import time
import math
import Pyro4
# import pickle as serialization
import serpent as serialization
from random import uniform, seed
from PIL import Image, ImageDraw
import cv2
import logging
import operator
import numpy as np
from collections import namedtuple
from extract import process_image


CONTROL_HOSTNAME = 'localhost'
# CONTROL_HOSTNAME = '192.168.0.27'

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.DEBUG)
log = logging.getLogger(__name__)


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
        s = process_image(f)
        return s


class Stone(object):

    def __init__(self, id, center, size, angle, color, structure):
        self.id = id
        self.center = center
        self.size = size
        self.angle = angle
        self.color = color
        self.structure = structure
        ux = size[0] * math.cos(math.radians(angle))
        uy = size[0] * math.sin(math.radians(angle))
        vx = size[1] * math.cos(math.radians(angle + 90))
        vy = size[1] * math.sin(math.radians(angle + 90))
        self.w = math.sqrt(ux ** 2 + vx ** 2)
        self.h = math.sqrt(uy ** 2 + vy ** 2)
        self.bbox = (center[0] - self.w, center[1] - self.h, center[0] + self.w, center[1] + self.h)

    def overlaps(self, stone):
        return (abs(self.center[0] - stone.center[0]) < (self.w + stone.w)) and (abs(self.center[1] - stone.center[1]) < (self.h + stone.h))

class StoneMap(object):

    def __init__(self, filename):
        self.filename = filename
        self.stones = {}
        self.size = 1000, 1000 # TODO: should be: 4000, 2000
        self.workarea = None
        try:
            with open(self.filename, 'rb') as f:
                d = serialization.load(f)
                self.stones = d['stones']
                self.size = d['size']
                self.workarea = d['workarea']
        except:
            self.save()

    # populate the map with random stones
    def randomize(self, count=2000):
        log.debug('Generating random map of %d stones', count)
        self.stones = {}
        for i in range(count):
            center = (uniform(50, self.size[0] - 50), uniform(50, self.size[1] - 50))
            a = uniform(10, 30)
            b = uniform(10, 30)
            if b > a:
                a, b = b, a
            r = uniform(-90, 90)
            c, d = ((uniform(96, 160), uniform(96, 160), uniform(96, 160)), uniform(10, 40))
            s = Stone(i, center, (a, b), r, c, d)
            bad = False
            for s2 in self.stones.values():
                if s.overlaps(s2):
                    bad = True
                    break
            if not bad:
                self.add_stone(s)
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

        # method #3
        def mrp(a):
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


        """
        # method #4 - has a bug :-(
        def mrp(a):
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
        """

        log.debug('Finding workarea')
        scale = 10 # work on less precise scale
        usage = np.zeros((int(math.ceil(self.size[0]/scale)), int(math.ceil(self.size[1]/scale))), dtype=np.bool)
        # calculate usage
        for s in self.stones.values():
            a, b, c, d = s.bbox
            for x in range(int(math.floor(a/scale)), int(math.floor(c/scale) + 1)):
                for y in range(int(math.floor(b/scale)), int(math.floor(d/scale) + 1)):
                    usage[x][y] = True
        # find workarea
        ll, ur = mrp(usage)
        wa = (ll.X * scale, ll.Y * scale, ur.X * scale, ur.Y * scale)
        return wa

    def image(self):
        log.debug('Creating map image')
        im = Image.new('RGB', self.size)
        draw = ImageDraw.Draw(im)
        for s in self.stones.values():
            draw.rectangle(s.bbox, outline='blue')
            size = int(math.ceil(s.size[0] * 2.0)), int(math.ceil(s.size[1] * 2.0))
            t = Image.new('RGBA', size)
            ImageDraw.Draw(t).ellipse(((0, 0), size), fill='white')
            t = t.rotate(s.angle, resample=Image.BILINEAR, expand=True)
            draw.bitmap((s.center[0] - t.size[0] / 2.0, s.center[1] - t.size[1] / 2.0), t)
        if self.workarea:
            draw.rectangle(self.workarea, outline='red')
        im.save('map.png')

    # functions also as replace
    def add_stone(self, stone):
        self.stones[stone.id] = stone

    def save(self):
        with open(self.filename, 'wb') as f:
            d = {'stones': self.stones, 'size': self.size, 'workarea': self.workarea}
            serialization.dump(d, f)


class Brain(object):

    def __init__(self):
        self.machine = Machine(CONTROL_HOSTNAME)
        self.map = StoneMap('stones.data')
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

        def dist(a, b):
            d = np.array(a) - np.array(b)
            return np.sqrt(np.dot(d, d))

        def similarity_index(s1, s2):
            dc = dist(s1.center, s2.center)
            ds = dist(s1.size, s2.size)
            da = abs(s1.angle - s2.angle)
            if dc > 20:
                return 0.0
            if ds > 20:
                return 0.0
            if da > 20:
                return 0.0
            return 1.0 - max([dc / 20.0, ds / 20.0, da / 20.0])

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
                s = simil_index(stones[i], stones[j])
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


    def demo_random_map(self):
        self.map.randomize()
        self.map.image()

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
    # brain.run('demo_random_map')
    # brain.run('scan')

