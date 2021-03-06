#!/usr/bin/python
import time
import math
import subprocess
import threading

import Pyro4
import cv2

from art import art_step
from extract import process_image

from utils import random_on_circle
from stone import StoneMap, Stone

from log import makelog
log = makelog('brain')

# CONTROL_HOSTNAME = 'localhost'
CONTROL_HOSTNAME = '192.168.0.29'


class Machine(object):
    """ High-level operations on CNC """

    def __init__(self, hostname):
        self.uri = 'PYRO:control@{}:5001'.format(hostname)
        self.control = Pyro4.Proxy(self.uri)
        self.cam = Camera(self)
        self.x, self.y, self.z, self.e = 0.0, 0.0, 0.0, 0.0
        self.last_pickup_height = None
        try:
            self.control.reset()
        except:
            log.warn('Could not initialize/reset machine')
            self.control = None

    def go(self, x=None, y=None, z=None, e=None):
        self.control.go(x=x, y=y, z=z, e=e)
        self.control.block()
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if z is not None:
            self.z = z
        if e is not None:
            self.e = e

    def check_movement(self, x=None, y=None, z=None, e=None):
        return self.control.check_movement(x=x, y=y, z=z, e=e)

    def lift_up(self, x, y, tries=5, jitter_rad=5):
        if self.last_pickup_height is not None:
            raise Exception('lift_up called, but previous call not cleared using lift_down')

        jit_x = 0
        jit_y = 0

        # try lifting up tries times
        for i in range(tries):
            log.info('Pickup try {} of {}'.format(i + 1, tries))
            self.control.light(True)
            # self.control.vacuum(True)   # Turned on in pickup routine
            h = self.control.pickup_custom()
            self.control.light(False)
            if h is not None:
                log.info('Picked up at height {}'.format(h, ))
                self.last_pickup_height = h
                return True

            self.go(x=x + jit_x, y=y + jit_y)
            jit_x, jit_y = random_on_circle(jitter_rad)
            self.control.block()

        log.info('Pickup failed after {} tries'.format(tries, ))
        return False

    def lift_down(self, extra_z_down=3.0):
        if self.last_pickup_height is None:
            raise Exception('lift_down called without calling lift_up first')
        self.control.light(True)
        self.go(z=max(self.last_pickup_height - extra_z_down, 0))
        self.control.vacuum(False)
        self.control.light(False)
        self.control.pickup_top()
        self.last_pickup_height = None

    def head_delta(self, angle=None):
        # length of rotating head (in mm)
        length = 40.0
        if angle is None:
            angle = self.e
        angle = math.radians(angle)
        return (0.0 + length * math.sin(angle), 0.0 + length * math.cos(angle))


class Camera(object):

    def __init__(self, machine, index=0):
        self.machine = machine
        self.index = index
        self.videodev = '/dev/video' + str(index)
        self.resx = 720.0  # image width (in pixels). Transposed!
        self.resy = 1280.0  # image height (in pixels). Transposed!
        self.viewx = 39.0 * 2.0  # view width (in cnc units = mm). Transposed!
        self.viewy = 69.0 * 2.0  # view height (in cnc units = mm). Transposed!
        self.flipall = True
        subprocess.call(['v4l2-ctl', '-d', self.videodev, '-c', 'white_balance_temperature_auto=0'])
        subprocess.call(['v4l2-ctl', '-d', self.videodev, '-c', 'white_balance_temperature=4667'])
        subprocess.call(['v4l2-ctl', '-d', self.videodev, '-c', 'exposure_auto=1'])  # means disable
        subprocess.call(['v4l2-ctl', '-d', self.videodev, '-c', 'exposure_absolute=39'])
        subprocess.call(['v4l2-ctl', '-d', self.videodev, '-c', 'brightness=30'])

    def pos_to_mm(self, pos, offset=(0, 0)):
        """ Calculate distance of perceived pixel from center of the view (in cnc units = mm) """
        dx, dy = -3.0, +66.00  # distance offset from head center to camera center
        x = dx + self.viewx * (pos[0] / self.resx - 0.5) + offset[0]
        y = dy + self.viewy * (pos[1] / self.resy - 0.5) + offset[1]
        return x, y

    def size_to_mm(self, size):
        """ Calculate size of perceived pixels (in cnc units = mm) """
        w = self.viewx * size[0] / self.resx
        h = self.viewy * size[1] / self.resy
        return w, h

    def grab(self, save=False):
        log.debug('Taking picture at coords {},{}'.format(self.machine.x, self.machine.y))
        self.machine.control.light(True)
        try:
            cam = cv2.VideoCapture(self.index)
            cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 720)
            cam.set(cv2.cv.CV_CAP_PROP_FPS, 10)
            # cam.set(cv2.cv.CV_CAP_PROP_EXPOSURE, 19)
            # cam.set(cv2.cv.CV_CAP_PROP_BRIGHTNESS, 10)

            for i in range(3):  # Dummy captures to clear the buffer
                cam.read()

            ret, frame = cam.read()

            cam.release()
            if ret:
                frame = cv2.transpose(frame)
                if self.flipall:
                    frame = cv2.flip(frame, -1)
                ret = frame
        except:
            ret = None
        self.machine.control.light(False)
        if ret is not None and save:
            fn = 'grab_{:04d}_{:04d}'.format(int(self.machine.x), int(self.machine.y))
            log.debug('Saving {}.jpg'.format(fn))
            cv2.imwrite('map/{}.jpg'.format(fn), ret)
        return ret

    def grab_extract(self, x, y, img=None, save=False):
        if img is None:
            frame = self.grab()
        else:
            frame = img

        if frame is None:
            log.warn('Failed to grab the image')
            return []
        fn = 'grab_{:04d}_{:04d}'.format(int(x), int(y))
        if save:
            log.debug('Saving {}.jpg'.format(fn))
            cv2.imwrite('map/{}.jpg'.format(fn), frame)
        stones, result_image, thresh_image, weight_image = process_image(fn, frame, save_stones='png')
        if save:
            log.debug('Saving {}-processed.jpg'.format(fn))
            cv2.imwrite('map/{}-processed.jpg'.format(fn), result_image)
            cv2.imwrite('map/{}-threshold.jpg'.format(fn), thresh_image)
            cv2.imwrite('map/{}-weight.jpg'.format(fn), weight_image * 255)
        log.debug('Found {} stones'.format(len(stones)))
        return stones


def save_map(map):
    log.debug('Saving map...')
    map.save()
    log.debug('Saving map. Done.')


class Brain(object):

    def __init__(self, use_machine=True):
        if use_machine:
            self.machine = Machine(CONTROL_HOSTNAME)
        else:
            self.machine = None

        self.map = StoneMap('stonemap')
        # shortcuts for convenience

        if self.machine:
            self.m = self.machine
            self.c = self.machine.control
            self.c.reset()
            self.c.home()

            # go home (also test if the machine is initialized and working)
            # self.c.home()

    def start(self):
        pass

    def run(self, prgname):
        f = getattr(self, prgname)
        f()

    def scan(self, startx=0, starty=0, analyze=True):
        log.debug('Begin scanning')
        self.c.pickup_top()
        self.z = self.c.get_pickup_z()
        self.m.go(e=90)
        self.c.block()

        stones = []
        x, y = self.map.size
        stepx = int(self.machine.cam.viewx / 2.0)
        stepy = int(self.machine.cam.viewy / 2.0)
        for i in range(int(startx), x + 1, stepx):
            for j in range(int(starty), y + 1, stepy):
                self.m.go(x=i, y=j)
                self.c.block()
                if analyze:
                    st = self.machine.cam.grab_extract(i, j, save=True)
                    for s in st:
                        s.center = self.machine.cam.pos_to_mm(s.center, offset=(i, j))
                        s.size = self.machine.cam.size_to_mm(s.size)
                        s.rank = 0.0
                        stones.append(s)
                else:
                    self.machine.cam.grab(save=True)
        log.debug('End scanning')
        if analyze:
            # select correct stones
            log.debug('Begin selecting/reducing stones')
            for i in range(len(stones)):
                for j in range(i + 1, len(stones)):
                    s = stones[i].similarity(stones[j])
                    stones[i].rank += s
                    stones[j].rank -= s
            log.debug('End selecting/reducing stones')
            # copy selected stones to storage
            self.map.stones = [s for s in stones if s.rank >= 1.5]
            self.map.save()

    def scan_from_files(self, analyze=True):
        import re
        import os
        import fnmatch
        import pickle as serialization

        log.debug('Begin scanning')

        cam = Camera(None)

        stones = []
        p = "map_offline/"   # looks here for pngs...

        pngfiles = []

        bins = []

        for file in os.listdir(p):
            if fnmatch.fnmatch(file, 'grab*.jpg'):
                pngfiles.append(file)

        for fn in pngfiles:
            m = re.search('\w+_(\d+)_(\d+)', fn)

            image = cv2.imread(os.path.join(p, fn), -1)
            (h, w) = image.shape[:2]

            xp, yp = float(m.group(1)), float(m.group(2))

            log.info('Reading file at {} {}'.format(xp, yp))

            localstones = []

            st = cam.grab_extract(xp, yp, img=image, save=False)
            for s in st:
                s.center = cam.pos_to_mm(s.center, offset=(xp, yp))
                s.size = cam.size_to_mm(s.size)
                s.rank = 0.0
                stones.append(s)
                localstones.append(s)

            bin = {'x': xp, 'y': yp, 'stones': localstones}
            bins.append(bin)

        with open('map/intermediate.data', 'wb') as f:
            serialization.dump(bins, f)

        log.debug('Found {} stones in {} bins'.format(len(stones), len(bins)))

        log.debug('End scanning')
        chosen_stones = []
        if analyze:
            # select correct stones
            log.debug('Begin selecting/reducing stones')
            for bi in range(len(bins)):
                bin_a = bins[bi]
                stones_a = bin_a['stones']

                other_stones = []

                for bj in range(bi + 1, len(bins)):
                    bin_b = bins[bj]
                    stones_b = bin_b['stones']

                    d_x = abs(bin_b['x'] - bin_a['x'])
                    d_y = abs(bin_b['y'] - bin_a['y'])

                    if d_x > cam.viewx * 2.0 or d_y > cam.viewy * 2.0:
                        continue

                    other_stones += stones_b

                for this_stone in stones_a:
                    for other_stone in other_stones:
                        if this_stone.coincides(other_stone):
                            other_stone.bogus = True

                    if not this_stone.bogus:
                        chosen_stones.append(this_stone)

            log.debug('End selecting/reducing stones')
            # copy selected stones to storage
            self.map.stones = [s for s in chosen_stones]
            self.map.stage = (0, 2, None, None)

            log.debug('Reduced from {} to {} stones'.format(len(stones), len(self.map.stones)))

            self.map.save(meta=True)

    def demo1(self):
        # demo program which moves stone back and forth
        while True:
            self._move_stone_absolute((3500, 1000),  0, (3500, 1250), 90)
            self._move_stone_absolute((3500, 1250), 90, (3500, 1000),  0)

    def demo2(self):
        while True:
            self._move_stone((3500, 1000),  30, (3500, 1000), 120)
            self._move_stone((3500, 1000), 120, (3500, 1000),  30)

    def demo3(self):
        while True:
            self._move_stone((3700, 1000),  30, (3700, 1000), 120)
            self._move_stone((3700, 1000), 120, (3700, 1000),  30)

    def _move_stone_absolute(self, c1, a1, c2, a2):
        log.debug('Abs moving stone center %s angle %s to center %s angle %s', str(c1), str(a1), str(c2), str(a2))

        if not self.m.check_movement(x=c1[0], y=c1[1], e=a1):
            log.warn('Invalid pickup position {},{}. Aborting move.'.format(c1[0], c1[1]))
            return False

        if not self.m.check_movement(x=c2[0], y=c2[1], e=a2):
            log.warn('Invalid placement position {},{}. Aborting move.'.format(c2[0], c2[1]))
            return False

        self.m.go(x=c1[0], y=c1[1], e=a1)
        ret = self.m.lift_up(x=c1[0], y=c1[1])

        if ret:
            self.m.go(x=c2[0], y=c2[1], e=a2)
            self.m.lift_down()
            return True
        else:
            return False

    def _turn_stone_calc(self, c1, sa, c2, ea):
        h1 = self.machine.head_delta(angle=sa)
        c1 = c1[0] - h1[0], c1[1] - h1[1]
        h2 = self.machine.head_delta(angle=ea)
        c2 = c2[0] - h2[0], c2[1] - h2[1]

        off = (0.0, 0.0)   # was 6.0, 0
        return (c1[0] + off[0], c1[1] + off[1]), (c2[0] + off[0], c2[1] + off[1])  # FIXME: Offseted

    def _move_stone(self, c1, a1, c2, a2):
        log.debug('Moving stone center %s angle %s to center %s angle %s', str(c1), str(a1), str(c2), str(a2))
        da = a1 - a2
        if da < 0.0:
            da = 360.0 + da
        da = da % 180

        # Case 1
        nc1, nc2 = self._turn_stone_calc(c1, 0.0, c2, da)

        max_y = self.map.size[1]
        if c1[1] >= 0 and c2[1] >= 0 and c1[1] <= max_y and c2[1] <= max_y:
            return self._move_stone_absolute(nc1, 0, nc2, da)
        else:   # Case 2
            nc1, nc2 = self._turn_stone_calc(c1, 180.0, c2, da)
            return self._move_stone_absolute(nc1, 180.0, nc2, da)
        # TODO: save map here ?

    def save_map(self):
        log.debug('Saving map...')
        self.map.save()
        log.debug('Saving map. Done.')

    def performance(self):
        saving_thread = threading.Thread(target=save_map, args=(self.map, ))

        while True:
            log.debug('Thinking...')
            i, nc, na, stage, force = art_step(self.map)

            if i is not None:
                s = self.map.stones[i]

                if nc is None:
                    nc = s.center

                if na is None:
                    na = s.angle

                log.debug('Placing stone {} from {} to {}'.format(i, s.center, nc))

                if self._move_stone(s.center, s.angle, nc, na):   # Pickup worked
                    if saving_thread.is_alive(): saving_thread.join() # wait until save is completed if still being done
                    s.center = nc
                    s.angle = na
                    self.map.stage = stage  # Commit stage
                    log.info('Placement worked')
                else:  # Fail, flag
                    if saving_thread.is_alive(): saving_thread.join() # wait until save is completed if still being done
                    s.flag = True
                    log.info('Placement failed')

                saving_thread = threading.Thread(target=save_map, args=(self.map, ))
                saving_thread.start() # async call of self.save_map

            elif force:  # Art wants us to advance anyhow
                if saving_thread.is_alive(): saving_thread.join() # wait until save is completed if still being done
                self.map.stage = stage  # Commit stage

                saving_thread = threading.Thread(target=save_map, args=(self.map, ))
                saving_thread.start() # async call of self.save_map
            else:
                if saving_thread.is_alive(): saving_thread.join() # wait until save is completed if still being done
                time.sleep(1)

        if saving_thread.is_alive(): saving_thread.join() # wait until save is completed if still being done

if __name__ == '__main__':
    brain = Brain(use_machine=True)
    brain.start()
    # brain.scan_from_files()
    # brain.scan(startx=2769, analyze=False)
    # brain.demo1()
    brain.performance()
