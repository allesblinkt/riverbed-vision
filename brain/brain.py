#!/usr/bin/env python3
import time
import math
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor

import Pyro4
import cv2

import config

from art import art_step
from extract import process_image

from utils import random_on_circle, frange_inclusive
from stone import StoneMap, Stone

from log import makelog
log = makelog('brain')

executor_save = ThreadPoolExecutor(max_workers=1)


class DummyMachine(config.Machine):

    def __init__(self, hostname=config.Machine.CONTROL_HOSTNAME):
        self.uri = 'PYRO:control@{}:5001'.format(hostname)
        log.info('Connecting to control server at %s', self.uri)

        self.control = Pyro4.Proxy(self.uri)
        self.cam = None  # Camera(self)
        self.x, self.y, self.z, self.e = 0.0, 0.0, 0.0, 0.0


class Machine(config.Machine):
    """ High-level operations on CNC """

    def __init__(self, hostname=config.Machine.CONTROL_HOSTNAME):
        self.uri = 'PYRO:control@{}:5001'.format(hostname)
        log.info('Connecting to control server at %s', self.uri)

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

    def lift_up(self, x, y, tries=config.Machine.lift_up_tries, jitter_rad=config.Machine.lift_up_jitter_rad):
        if self.last_pickup_height is not None:
            raise Exception('lift_up called, but previous call not cleared using lift_down')

        jit_x, jit_y = 0, 0

        # try lifting up tries times
        for i in range(tries):
            log.info('Pickup try {} of {}'.format(i + 1, tries))
            # self.control.vacuum(True)   # Turned on in pickup routine
            h = self.control.pickup_custom()
            if h is not None:
                log.info('Picked up at height {}'.format(h, ))
                self.last_pickup_height = h
                return True

            self.go(x=x + jit_x, y=y + jit_y)
            jit_x, jit_y = random_on_circle(jitter_rad)
            self.control.block()

        log.info('Pickup failed after {} tries'.format(tries, ))
        return False

    def lift_down(self, extra_z_down=config.Machine.lift_down_extra_z_down):
        if self.last_pickup_height is None:
            raise Exception('lift_down called without calling lift_up first')
        self.go(z=max(self.last_pickup_height - extra_z_down, 0))
        self.control.vacuum(False)
        self.control.eject(True)
        self.control.dwell(self.lift_down_eject_dwell)
        self.control.eject(False)
        self.control.pickup_top()
        self.last_pickup_height = None

    def head_delta(self, angle=None):
        # length of rotating head (in mm)
        if not self.head_arm_length:
            return (0.0, 0.0)
        if angle is None:
            angle = self.e
        angle = math.radians(angle)
        return (0.0 + self.head_arm_length * math.sin(angle), 0.0 + self.head_arm_length * math.cos(angle))


class Camera(config.Camera):

    def __init__(self, machine, index=0):
        self.machine = machine
        self.index = index
        self.videodev = '/dev/video' + str(index)

        if self.view_transpose:
            self.resx = self.capture_height
            self.resy = self.capture_width
        else:
            self.resx = self.capture_width
            self.resy = self.capture_height

        log.info('Video device %s', self.videodev)

        cam = cv2.VideoCapture(self.index)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.capture_width))
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.capture_height))
        cam.set(cv2.CAP_PROP_FPS, 10)
        # cam.set(cv2.CAP_PROP_EXPOSURE, 19)
        # cam.set(cv2.CAP_PROP_BRIGHTNESS, 10)
        for i in range(self.grab_dummy_frames):
            cam.read()

        time.sleep(0.5)  # Wait for cam to be ready

        v4l_cmd = ['v4l2-ctl', '-d', str(index)]
        for param_key, param_val in self.v4l_params_1.items():
            cmd_params = ['-c', '%s=%d' % (param_key, param_val)]
            subprocess.call(v4l_cmd + cmd_params)
        time.sleep(0.5)
        for param_key, param_val in self.v4l_params_2.items():
            cmd_params = ['-c', '%s=%d' % (param_key, param_val)]
            subprocess.call(v4l_cmd + cmd_params)

        self.cam = cam

    def set_cam_parameter(self, key, value):
        v4l_cmd = ['v4l2-ctl', '-d', str(self.index)]
        cmd_params = ['-c', '%s=%d' % (key, value)]
        subprocess.call(v4l_cmd + cmd_params)

    def pos_to_mm(self, pos, offset=(0, 0)):
        """ Calculate distance of perceived pixel from center of the view
            (in cnc units = mm) """
        # distance offset from head center to camera center
        dx = self.offset_x
        dy = self.offset_y
        x = dx + self.viewx * (pos[0] / self.resx - 0.5) + offset[0]
        y = dy + self.viewy * (pos[1] / self.resy - 0.5) + offset[1]
        return x, y

    def camera_pos_to_mm(self, pos):
        """ Converts from camera center to head coordinates
            In other words, if you want to have the camera over a specific
            point, you would use this... """
        # distance offset from head center to camera center
        dx = self.offset_x
        dy = self.offset_y
        x = pos[0] - dx
        y = pos[1] - dy
        return x, y

    def camera_center_to_mm(self, pos):
        """ Calculate where the camera center is at a given head position """
        return self.pos_to_mm((self.resx * 0.5, self.resy * 0.5), offset=(pos[0], pos[1]))

    def size_to_mm(self, size):
        """ Calculate size of perceived pixels (in cnc units = mm) """
        w = self.viewx * size[0] / self.resx
        h = self.viewy * size[1] / self.resy
        return w, h

    def grab(self, save=False, light_channel=None, suffix=''):
        log.debug('Taking picture at coords {},{} (light_channel={})'.format(self.machine.x, self.machine.y, light_channel))
        cam = self.cam
        self.machine.control.light(True, light_channel)

        try:
            for i in range(self.grab_dummy_frames):
                cam.read()

            ret, frame = cam.read()

            if ret:
                if self.view_transpose:
                    frame = cv2.transpose(frame)
                if self.view_flip:
                    frame = cv2.flip(frame, -1)
                ret = frame
        except:
            ret = None
        self.machine.control.light(False, light_channel)
        if ret is not None and save:
            if light_channel is not None:
                fn = 'grab_{:04d}_{:04d}_l{:d}{}'.format(int(self.machine.x), int(self.machine.y), light_channel, suffix)
            else:
                fn = 'grab_{:04d}_{:04d}{}'.format(int(self.machine.x), int(self.machine.y), suffix)
            log.debug('Saving {}.jpg'.format(fn))
            cv2.imwrite('map/{}.jpg'.format(fn), ret)
        return ret

    def grab_light_sequence(self, save=False):
        self.grab(save=save, light_channel=0)
        self.grab(save=save, light_channel=1)
        self.grab(save=save, light_channel=2)
        self.grab(save=save, light_channel=3)

    def grab_focus_sequence(self, save=False):
        res = []
        for focus in self.focus_stack:
            self.set_cam_parameter('focus_absolute', focus)
            suffix = '_f%d' % (focus, )
            im = self.grab(save=save, light_channel=None, suffix=suffix)
            res.append(im)
        return res

    def grab_extract(self, x, y, img=None, save=False, double_focus=False):
        if img is None:
            if double_focus:
                frame = self.grab_focus_sequence()
            else:
                frame = self.grab()
        else:
            frame = img

        if frame is None:
            log.warn('Failed to grab the image')
            return []
        fn = 'grab_{:04d}_{:04d}'.format(int(x), int(y))
        if save:
            if double_focus:
                log.debug('Saving {}_f30.jpg'.format(fn))
                cv2.imwrite('map/{}.jpg'.format(fn), frame[0])
                log.debug('Saving {}_f60.jpg'.format(fn))
                cv2.imwrite('map/{}.jpg'.format(fn), frame[1])
            else:
                log.debug('Saving {}.jpg'.format(fn))
                cv2.imwrite('map/{}.jpg'.format(fn), frame)

        if self.machine:
            self.machine.control.light(True)

        stones, result_image, thresh_image, weight_image = process_image(fn, frame, save_stones='png')
        
        if self.machine:
            self.machine.control.light(False)

        if save:
            log.debug('Saving {}-processed.jpg'.format(fn))
            cv2.imwrite('map/{}-processed.jpg'.format(fn), result_image)
            cv2.imwrite('map/{}-threshold.jpg'.format(fn), thresh_image)
            cv2.imwrite('map/{}-weight.jpg'.format(fn), weight_image * 255)

        log.debug('Found {} stones'.format(len(stones)))
        return stones


class Brain(config.Brain):

    def __init__(self, machine=None, create_new_map=False):
        self.machine = machine
        self.stone_map = StoneMap('stonemap', create_new=create_new_map)

        # self.stone_map.stage = (0, 3, None, None, None)  # NOTE: use this to override the current state


        self.cycle_count = 0   # How many cycles in this run
        self.next_break_cycle = self.lunch_break_every[0]   # When to lunch break next?

        if self.machine:
            # shortcuts for convenience
            self.m = self.machine
            self.c = self.machine.control
            self.c.reset()
            self.c.home()
            self.c.go(x=self.init_x, y=self.init_y, e=self.init_e)
            self.c.feedrate(self.init_feedrate)

    def run(self, prgname):
        f = getattr(self, prgname)
        f()

    def scan_update(self, stone):
        self.m.go(e=self.init_e)

        log.debug('Continous scan: start (%d stones)', len(self.stone_map.stones))
        x, y = self.machine.x, self.machine.y
        st = self.machine.cam.grab_extract(x, y, save=False, double_focus=True)
        new_stones = []

        for s in st:
            s.center = self.machine.cam.pos_to_mm(s.center, offset=(x, y))
            s.size = self.machine.cam.size_to_mm(s.size)
            s.rank = 0.0
            s.bogus = False
            if self.stone_map.is_inside(s):
                new_stones.append(s)

        log.debug('Continous scan: found %d stones', len(new_stones))

        camera = self.machine.cam

        camera_center = camera.camera_center_to_mm((x, y))
        old_stones = self.stone_map.get_at_with_border(camera_center, border_size=max(camera.resx, camera.resy))

        self.save_map_wait()

        add_count = 0
        remove_count = 0
        purge_count = 0
        the_new_stone = None

        backup_map = self.stone_map.copy()

        # Remove doublettes
        for new_stone in new_stones:
            # Add it
            add_count += 1
            self.stone_map.add_stone(new_stone)

            for old_stone in old_stones:
                if new_stone.coincides(old_stone):
                    if self.stone_map.remove_stone(old_stone):
                        # carry over flags if the stone is not the center one
                        if old_stone != stone:
                            new_stone.flag = old_stone.flag
                        # else mark the stone as new one
                        else:
                            the_new_stone = new_stone
                        remove_count += 1

        # Purge
        for old_stone in old_stones:
            cx = old_stone.center[0]
            cy = old_stone.center[1]

            d_x = abs(camera_center[0] - cx)
            d_y = abs(camera_center[1] - cy)

            stone_ext = old_stone.size[0]

            bounds_tol = 0.5 * 0.75 * 0.5
            center_tol = 0.5 * 0.5
            is_in_bounds = d_x + stone_ext < camera.viewx * bounds_tol and d_y + stone_ext < camera.viewy * bounds_tol
            is_in_center = d_x < camera.viewx * center_tol and d_y < camera.viewy * center_tol
            if is_in_bounds and is_in_center:
                if self.stone_map.remove_stone(old_stone):
                    purge_count += 1

        if backup_map.stone_count() - self.stone_map.stone_count() >= 2:
            self.stone_map = backup_map
            log.debug('Ignored addition of %d new stones and removes %d old stones. %d purged', add_count, remove_count, purge_count)
            return stone
        else:
            log.debug('Added %d new stones and removes %d old stones. %d purged', add_count, remove_count, purge_count)
            return the_new_stone

        # select stones outside of the view
        # TODO: get these right:

        # borderx, bordery = 0, 0
        # stones_o = [s for s in self.stone_map.stones if abs(s.center[0] - x) > (self.machine.cam.viewx / 2.0 - borderx) and abs(s.center[1] - y) > (self.machine.cam.viewy / 2.0 - bordery)]
        # log.debug('Continous scan: removing %d stones', len(self.stone_map.stones) - stones_o)
        # self.stone_map.stones = stones_o + stones_n
        # log.debug('Continous scan: end (%d stones)', len(self.stone_map.stones))
        # self.stone_map.save()

    def scan(self, startx=0, starty=0, analyze=True):
        log.debug('Begin scanning')
        self.c.scan_top()
        self.z = self.c.get_scan_z()
        self.m.go(e=self.init_e)
        self.c.block()

        stones = []
        end_x, end_y = self.stone_map.size
        step_x = int(self.machine.cam.viewx * self.scan_step[0])
        step_y = int(self.machine.cam.viewy * self.scan_step[1])

        for x in frange_inclusive(int(startx), end_x, step_x):
            for y in frange_inclusive(int(starty), end_y, step_y):
                self.m.go(x=x, y=y)
                self.c.block()

                log.info('Scanning at x: %d y: %d', x, y)

                if analyze:
                    st = self.machine.cam.grab_extract(x, y, save=True, double_focus=True)
                    for s in st:
                        s.center = self.machine.cam.pos_to_mm(s.center, offset=(x, y))
                        s.size = self.machine.cam.size_to_mm(s.size)
                        s.rank = 0.0
                        if self.stone_map.is_inside(s):
                            stones.append(s)
                else:
                    self.machine.cam.grab_focus_sequence(save=True)
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
            self.stone_map.stones = [s for s in stones if s.rank >= 1.5]
            self.stone_map.save()

    def scan_from_files(self, analyze=True, picdir='map'):
        import re
        import os
        import fnmatch
        import pickle as serialization

        log.debug('Begin scanning')

        cam = Camera(None)

        stones = []
        pngfiles = []
        bins = []

        for file in sorted(os.listdir(picdir)):
            if fnmatch.fnmatch(file, 'grab*_f30.jpg'):
                pngfiles.append(file)

        pngfiles = pngfiles[::-1]   # reverse, cause trouble is likely to be at the end
        # pngfiles = pngfiles[:100]   # reverse, cause trouble is likely to be at the end

        for fn in pngfiles:
            log.info('Checking file {}'.format(fn, ))

            m = re.search('\w+_(\d+)_(\d+)_f30', fn)

            image = cv2.imread(os.path.join(picdir, fn), -1)
            image2 = cv2.imread(os.path.join(picdir, fn.replace('_f30', '_f60')), -1)

            (h, w) = image.shape[:2]
            (h2, w2) = image2.shape[:2]

            del image, image2

        for fn in pngfiles:
            log.info('Starting file {}'.format(fn, ))

            m = re.search('\w+_(\d+)_(\d+)_f30', fn)

            image = cv2.imread(os.path.join(picdir, fn), -1)
            image2 = cv2.imread(os.path.join(picdir, fn.replace('_f30', '_f60')), -1)
            (h, w) = image.shape[:2]

            xp, yp = float(m.group(1)), float(m.group(2))

            log.info('Reading file at {} {}'.format(xp, yp))

            localstones = []

            st = cam.grab_extract(xp, yp, img=[image, image2], save=False, double_focus=True)

            del image, image2

            for s in st:
                s.center = cam.pos_to_mm(s.center, offset=(xp, yp))
                s.size = cam.size_to_mm(s.size)
                s.rank = 0.0
                if self.stone_map.is_inside(s):
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
                        if self.stone_map.is_inside(this_stone):
                            chosen_stones.append(this_stone)

            log.debug('End selecting/reducing stones')
            # copy selected stones to storage
            self.stone_map.stones = [s for s in chosen_stones]
            self.stone_map.stage = (0, 0, None, None, None)

            log.debug('Reduced from {} to {} stones'.format(len(stones), len(self.stone_map.stones)))

            self.stone_map.save(meta=True)

    def _move_stone_absolute(self, c1, a1, c2, a2):
        log.debug('Abs moving stone center %s angle %s to center %s angle %s', str(c1), str(a1), str(c2), str(a2))

        if not self.m.check_movement(x=c1[0], y=c1[1], e=a1):
            log.warn('Invalid pickup position {},{}. Aborting move.'.format(c1[0], c1[1]))
            return False

        if not self.m.check_movement(x=c2[0], y=c2[1], e=a2):
            log.warn('Invalid placement position {},{}. Aborting move.'.format(c2[0], c2[1]))
            return False

        self.m.go(x=c1[0], y=c1[1])
        self.c.light(True)
        self.c.dwell(1000)
        self.m.go(e=a1)
        ret = self.m.lift_up(x=c1[0], y=c1[1])
        self.c.light(False)

        if ret:
            self.m.go(x=c2[0], y=c2[1])
            self.c.light(True)
            self.c.dwell(1000)
            self.m.go(e=a2)
            self.m.lift_down()
            self.c.light(False)

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
        max_y = self.stone_map.size[1]
        if c1[1] >= 0 and c2[1] >= 0 and c1[1] <= max_y and c2[1] <= max_y:
            return self._move_stone_absolute(nc1, 0, nc2, da)
        else:   # Case 2
            nc1, nc2 = self._turn_stone_calc(c1, 180.0, c2, da)
            return self._move_stone_absolute(nc1, 180.0, nc2, da)
        # TODO: save map here ?

    def save_map_wait(self):
        # wait for map save if in progress
        if hasattr(self, 'future_save'):
            self.future_save.result()
            del self.future_save

    def save_map(self):
        self.save_map_wait()
        self.future_save = executor_save.submit(self.stone_map.save, meta=True, image=True)

    def next_step(self):
        log.debug('Getting next step...')
        r = art_step(self.stone_map)
        log.debug('Getting next step. Done.')
        return r

    def performance(self):

        while True:
            chosen_stone, nc, na, stage, force = self.next_step()

            log.info('Pausing here, if state paused...')
            self.c.check_pause()
            log.info('Continuing.')

            if self.cycle_count > self.next_break_cycle:
                log.info('Go home for lunchbreak time')
                self.c.scan_top()
                self.c.home_z()
                log.info('Making lunch break here, if speed slow...')
                self.c.check_lunch_break(sleep_s=self.lunch_break_duration)
                self.cycle_count += 1
                self.next_break_cycle += random.randint(self.lunch_break_every[0], self.lunch_break_every[1])
                log.info('Continuing.')
            else:
                self.cycle_count += 1

            if chosen_stone is not None:
                s = chosen_stone
                if nc is None:
                    nc = s.center

                if na is None:
                    na = s.angle

                log.debug('Placing stone {} from {} to {}'.format(s, s.center, nc))

                success_move = self._move_stone(s.center, s.angle, nc, na)
                self.save_map_wait()

                if success_move:   # Pickup worked
                    self.stone_map.move_stone(s, new_center=nc, angle=na)
                    self.stone_map.stage = stage  # Commit stage

                    putdown_pos = (nc[0], nc[1])
                    scan_pos = self.machine.cam.camera_pos_to_mm(putdown_pos)
                    self.m.go(x=scan_pos[0], y=scan_pos[1], e=90)

                    # we don't scan here, but let's pretend we do
                    self.c.light(True)
                    self.c.dwell(1000)
                    self.c.light(False)

                    log.info('Placement worked')
                    self.save_map()
                else:  # Fail, flag
                    log.info('Placement failed')
                    s.flag = True

                    pickup_pos = (s.center[0], s.center[1])
                    scan_pos = self.machine.cam.camera_pos_to_mm(pickup_pos)
                    self.m.go(x=scan_pos[0], y=scan_pos[1], e=90)
                    new_s = self.scan_update(s)

                    if new_s and new_s != s:
                        # try pickup again
                        log.debug('Retry placing stone {} from {} to {}'.format(s, new_s.center, nc))
                        success_move = self._move_stone(new_s.center, new_s.angle, nc, na)
                        if success_move:
                            self.stone_map.move_stone(new_s, new_center=nc, angle=na)
                            self.stone_map.stage = stage  # Commit stage
                            log.info('Retry placement worked')
                            self.save_map()
                        else:
                            new_s.flag = True
                            log.info('Retry placement failed')

                    self.save_map()

            elif force:  # Art wants us to advance anyhow
                self.save_map_wait()
                self.stone_map.stage = stage  # Commit stage
                self.save_map()
            else:
                time.sleep(1)


def brain_scan(startx=0):
    m = Machine()
    brain = Brain(machine=m, create_new_map=True)
    brain.scan(startx=startx, analyze=False)


def brain_analyze_offline(picdir):
    brain = Brain(create_new_map=True)
    brain.scan_from_files(analyze=True, picdir=picdir)


def brain_performance():
    m = Machine()
    brain = Brain(machine=m, create_new_map=False)
    brain.performance()


def parse_args():
    """ Parse command line arguments and return args namespace """
    import argparse

    parser = argparse.ArgumentParser(description="Run Jllers brain")

    parser = argparse.ArgumentParser()
    command_group = parser.add_mutually_exclusive_group()
    command_group.add_argument('performance', help='Performs the usual business', nargs='?', default='performance')
    command_group.add_argument('scan', help='Scan the surface to images', nargs='?')
    command_group.add_argument('analyze', help='Analyze pictures offline and create initial map', nargs='?')

    parser.add_argument('--scan-start-x', default=0, type=int, help='Start the scan at specified X position')
    parser.add_argument('--analyze-dir', default='map_offline', type=str, help='Perform offline analysis from images in this directory')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    try:
        if args.performance == 'performance':
            log.exception('Doing performance!')
            brain_performance()
        elif args.performance == 'scan':
            log.exception('Doing scan from %d mm !', args.scan_start_x)
            brain_scan(startx=args.scan_start_x)
        elif args.performance == 'analyze':
            log.exception('Doing analyze from %s !', args.analyze_dir)
            brain_analyze_offline(args.analyze_dir)
        else:
            raise ValueError('Performance option not recognized')
    except:
        log.exception('exception in main:')
        raise
