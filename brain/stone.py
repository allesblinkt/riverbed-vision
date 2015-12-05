# import pickle as serialization
import serpent as serialization
from random import uniform
from collections import namedtuple
from PIL import Image, ImageDraw
import math

from log import log
from utils import *

class Stone(object):

    def __init__(self, id, center, size, angle, color, structure):
        self.id = id
        self.center = center
        if size[1] > size[0]:
            size = size[1], size[0]
            angle += 90
        self.size = size
        self.angle = angle % 180
        self.color = color
        self.structure = structure

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
            a, b = s.center[0] - s.size[0], s.center[1] - s.size[0]
            c, d = s.center[0] + s.size[0], s.center[1] + s.size[0]
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
