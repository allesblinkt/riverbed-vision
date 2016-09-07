#!/usr/bin/python

import pickle as serialization
# import serpent as serialization
from random import uniform

import cv2
import numpy as np
import time
import shutil

from utils import distance, SpatialHashMap
from art import art_step

from log import makelog
log = makelog(__name__)


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

    def overlaps(self, other):
        """ Checks whether stone overlaps with another stone """
        dx = self.center[0] - other.center[0]
        dy = self.center[1] - other.center[1]
        dsq = dx * dx + dy * dy
        rs = (self.size[0] + other.size[0] + 2)
        # d = distance(self.center, other.center)
        # TODO: double check all this...
        return dsq < rs * rs  # add 2 mm

    def coincides(self, stone):
        """ Check if two stones overlap so much that they could be the same """
        d = distance(self.center, stone.center)
        return d < (self.size[1] + stone.size[1]) * 0.5

    def similarity(self, stone):
        """ Computes similarity with another stone """
        dc = distance(self.center, stone.center)
        ds = distance(self.size, stone.size)
        da = abs(self.angle - stone.angle)

        threshold = 20.0

        if dc > threshold:
            return 0.0
        if ds > threshold:
            return 0.0
        if da > threshold:
            return 0.0
        return 1.0 - max([dc / threshold, ds / threshold, da / threshold])

    def save(self, filename):
        with open(filename, 'wb') as f:
            serialization.dump(self, f)


class StoneHole(object):

    def __init__(self, stone):
        self.center = stone.center
        self.size = min(stone.size)


class StoneMap(object):

    def __init__(self, name, create_new=False):
        self.name = name
        self.stones = []
        self.holes = []
        self.size = 3770, 1730
        self.stage = 0

        log.info('Loading map')

        # self.idx = index.Index()

        try:
            meta = False

            with open('map/{}.data'.format(self.name), 'rb') as f:
                log.debug('Opening map file')
                d = serialization.load(f)
                log.debug('Opened map file')

                self.size = d['size']

                if not isinstance(d['stage'], (int, float, complex)):
                    self.stage = d['stage']
                else:
                    self.stage = None

                if 'stones' in d:   # TODO: remove old format at some point?
                    log.debug('Loading stones from OLD format')
                    self.stones = d['stones']
                else:
                    log.debug('Loading stones from NEW format #1')
                    self.stones = d['stones1']
                    meta = True
            if meta:
                with open('map/{}.data2'.format(self.name), 'rb') as f:
                    # TODO: latin1 is just here because otherwise Py3 pickle cannot read stuff from Py2 pickle
                    # which contains numpy structures
                    d = serialization.load(f, encoding='latin1')
                    log.debug('Loading stones from NEW format #2')
                    sm = d['stones2']

                    assert len(self.stones) == len(sm)
                    for i in range(len(sm)):
                        self.stones[i].color = sm[i]['color']
                        self.stones[i].structure = sm[i]['structure']
        except Exception as e:
            log.warn('Something happened while loading')
            log.warn(e)

            if create_new:
                log.info('Creating map file')
                self.save(meta=True)
            else:
                raise(e)

        # Update spatialhashmap
        self.spatialmap = SpatialHashMap(cell_size=20)  # TODOL cell size to settings

        for stone in self.stones:
            self.spatialmap.insert_object_at_point(point=stone.center, obj=stone)

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

        # Manual, so we can resume with different scan
        self.maxstonesize = 22.0 * 2.0

    def is_inside(self, stone):
        """Check if a stone center is within the bounds of the map."""

        if stone.center[0] - stone.size[0] <= 0:
            return False
        if stone.center[1] - stone.size[0] <= 0:
            return False
        if stone.center[0] + stone.size[0] >= self.size[0]:
            return False
        if stone.center[1] + stone.size[0] >= self.size[1]:
            return False

        return True

    def get_at_with_border(self, center, border_size):
        return self.spatialmap.get_at_with_border(center, border_size)

    def can_put(self, stone):
        """Checks if we can put the stone to a new position."""
        border_size = (max(stone.size) + self.maxstonesize)  # FIXME: Check *2 or not. Not sure
        candidates = self.spatialmap.get_at_with_border(stone.center, border_size)

        return self.can_put_list(stone, candidates)

    def can_put_list(self, stone, stones):
        """Check if we can put a stone. Warning: Slow. Simply compares against the whole list of given stones."""
        if not self.is_inside(stone):
            return False

        # sr = 50
        for s in stones:
            if stone.overlaps(s):
                return False
        return True

    def move_stone(self, stone, new_center, angle=None):
        self.spatialmap.update_object_at_point(stone.center, new_center, stone)

        stone.center = new_center

        if angle is not None:
            stone.angle = angle

    def add_stone(self, stone):
        self.stones.append(stone)
        self.spatialmap.insert_object_at_point(stone.center, stone)

    def remove_stone(self, stone):
        if stone in self.stones:
            self.stones.remove(stone)
            self.spatialmap.remove(stone)

    def randomize(self, count=2000):
        """ Populate the map with random stones """

        log.debug('Generating random map of %d stones', count)
        self.stones = []   # TODO: update map
        failures = 0
        while count > 0 and failures < 100:
            center = (uniform(30, self.size[0] - 1000 - 30), uniform(30, self.size[1] - 30))
            a = uniform(6, 30)
            b = uniform(6, 20)
            if b > a:
                a, b = b, a
            r = uniform(-90, 90)
            c = uniform(0, 255), uniform(42, 226), uniform(20, 223)
            s = [uniform(0.001, 0.02) for i in range(40)] + [uniform(0.15, 0.3), uniform(0.2, 0.4)]
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
        """Draw the map to an image."""

        log.debug('Creating map image')
        for s in self.stones:
            center, size, angle = s.center, s.size, s.angle
            center = int((self.size[0] - center[0]) / scale), int(center[1] / scale)
            size = int(size[0] / scale), int(size[1] / scale)
            dummy = np.array([np.array([s.color], dtype=np.uint8)])
            color = (cv2.cvtColor(dummy, cv2.COLOR_LAB2BGR)[0, 0]).tolist()
            # structure = s.structure
            cv2.ellipse(img, center, size, 360 - angle, 0, 360, color, -1)

            if s.flag:
                cv2.circle(img, center, 3, (0, 69, 255))

        for h in self.holes:
            center = int((self.size[0] - h.center[0]) / scale), int(h.center[1] / scale)
            size = int(h.size / scale)
            cv2.circle(img, center, size, (255, 255, 255))

    def image_svg(self, svg, scale, stone_scale=1.0, metadata_scale=1.0, crosshair_scale=1.0, hole_scale=1.0):
        """Draw the map to an svg image."""

        import svgwrite

        dwg = svg
        log.debug('Creating map image svg')

        dwg.add(dwg.line((0, 0), (10, 0), stroke=svgwrite.rgb(10, 10, 16, '%')))
        dwg.add(dwg.text('Test', insert=(0, 0.2), fill='red'))

        stones = self.stones
        holes = self.holes

        for s in stones:
            center, size, angle = s.center, s.size, s.angle
            center = float((self.size[0] - center[0]) / scale), float(center[1] / scale)
            size = float(size[0] / scale) * stone_scale, float(size[1] / scale) * stone_scale
            dummy = np.array([np.array([s.color], dtype=np.uint8)])
            color = (cv2.cvtColor(dummy, cv2.COLOR_LAB2BGR)[0, 0]).tolist()
            svgcolor = svgwrite.rgb(color[2], color[1], color[0], 'RGB')
            structure = s.structure

            if stone_scale > 0.0:
                svg_ellip = dwg.ellipse(center=center, r=size, fill=svgcolor)
                svg_ellip.rotate(360 - angle, center=center)
                dwg.add(svg_ellip)

            if crosshair_scale > 0.0:
                cl = 2.0 * crosshair_scale

                svg_line1 = dwg.line(start=(center[0] + cl, center[1]), end=(center[0] - cl, center[1]), stroke='red')
                svg_line1.rotate(360 - angle, center=center)
                dwg.add(svg_line1)
                svg_line2 = dwg.line(start=(center[0], center[1] + cl), end=(center[0], center[1] - cl), stroke='red')
                svg_line2.rotate(360 - angle, center=center)
                dwg.add(svg_line2)

            if metadata_scale > 0.0:
                bar_w = (0.25 / scale) * metadata_scale
                bar_h = (20.0 / scale) * metadata_scale

                hist = structure[1:-2]
                num_bins = len(hist)

                for i in range(num_bins):
                    x = center[0] + i * bar_w
                    h = bar_h * hist[i] * 10.0
                    y = center[1] + (bar_h - h)

                    svg_rect = dwg.rect(insert=(x, y), size=(bar_w, h), fill='blue')
                    dwg.add(svg_rect)

            # if s.flag:
            #    cv2.circle(img, center, 3, (0, 69, 255))

        for h in holes:
            center = float((self.size[0] - h.center[0]) / scale), float(h.center[1] / scale)
            size = float(h.size / scale) * hole_scale

            if hole_scale > 0.0:
                # log.debug('Hole. Center: %s   Size: %f', center, size)
                svg_circle = dwg.circle(center=center, r=size, stroke='black', stroke_width=0.8, fill='none')

                dwg.add(svg_circle)

        # if self.workarea:
        #     a, b, c, d = self.workarea
        #     a, b, c, d = a / scale, b / scale, c / scale, d / scale
        #     cv2.rectangle(img, (a,b), (c,d), color=(255, 0, 0))

    def save(self, meta=False):
        log.debug('Saving map...')
        with open('map/{}.data'.format(self.name), 'wb') as f:
            s = [Stone(x.center, x.size, x.angle, None, None, x.flag) for x in self.stones]
            d = {'stones1': s, 'size': self.size, 'stage': self.stage}
            serialization.dump(d, f)
        ts = int(time.time())
        # backup with timestamp
        shutil.copy('map/{}.data'.format(self.name), 'map/{}-{}.data'.format(self.name, ts))
        if meta:
            with open('map/{}.data2'.format(self.name), 'wb') as f:
                s = [{'color': x.color, 'structure': x.structure} for x in self.stones]
                d = {'stones2': s}
                serialization.dump(d, f)
            # backup with timestamp
            shutil.copy('map/{}.data2'.format(self.name), 'map/{}-{}.data'.format(self.name, ts))
        log.debug('Saving map... Done.')

if __name__ == '__main__':
    map = StoneMap('stonemap')
    #map.stage = (0, 0, None, None)  # NOTE: use this to override the current state
    map.save(meta=True)


    if len(map.stones) == 0:
        # map.randomize()
        log.warn('No STONES!')

    while True:
        img_map = np.zeros((map.size[1] / 2, map.size[0] / 2, 3), np.uint8)
        map.image(img_map, 2)
        cv2.imshow('map', img_map)

        cvkey = chr(cv2.waitKey(1) & 255)
        if cvkey == 'q':
            break

        if cvkey == 's':
            import svgwrite

            ts = int(time.time())
            svg_fn = 'map_{}.svg'.format(ts, )

            log.debug('Saving svg to {}'.format(svg_fn))
            svg_drawing = svgwrite.Drawing(svg_fn)
            map.image_svg(svg_drawing, scale=2, stone_scale=0.89)
            svg_drawing.save()

        i, nc, na, stage, force = art_step(map)

        do_fail = False               # Never fail
        # do_fail = random() < 0.05   # Simulates that 5% of stones cannot be picked up

        if i is not None and not do_fail:
            stone = map.stones[i]

            map.holes.append(StoneHole(map.stones[i]))

            if nc is not None and na is not None and not do_fail:
                log.debug('Placing stone {} from {} to {}'.format(i, stone.center, nc))

                map.move_stone(stone, nc, na)

            map.stage = stage
        elif i is not None:
            stone = map.stones[i]
            stone.flag = True
        elif force:
            map.stage = stage

    map.save()
