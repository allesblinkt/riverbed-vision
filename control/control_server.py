#!/usr/bin/env python3
import re
import serial
import threading
import logging
import Pyro4
import netifaces
import time

import status

NEWLINE = b'\n'
DEVICE  = '/dev/ttyAMA0'
PORT    = 5001
IFACE   = 'wlan0'

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.DEBUG)
log = logging.getLogger(__name__)


def format_feed(**kwargs):
    return ' '.join(['{}{:0.1f}'.format(k.upper(), v) for k, v in sorted(kwargs.items()) if v is not None])


def format_pos(**kwargs):
    return ' '.join(['{}{:0.2f}'.format(k.upper(), v) for k, v in sorted(kwargs.items()) if v is not None])


@Pyro4.expose
class MachineController(object):

    def __init__(self, port_name, baudrate=19200):
        super(MachineController, self).__init__()

        self.serial_mutex = threading.Lock()
        self.serial_port = None
        self.port_name = port_name

        self.limits_x = [0, 3730]
        self.limits_y = [0, 1730]
        self.limits_z = [0, 110]
        self.limits_e = [-10000, 10000]

        self.scan_z = 100.0
        self.pickup_z = 40.0

        if port_name:
            log.info('Opening serial port %s at %d bps', port_name, baudrate)
            try:
                self.serial_port = serial.Serial(port_name)
                # self.serial_port.setTimeout(1.0)
                self.serial_port.baudrate = baudrate
                self.serial_port.write(NEWLINE)
                self.serial_port.write(NEWLINE)
                self.serial_port.write(NEWLINE)
            except SerialOpenError:
                raise SerialOpenError(port_name, baudrate)
        else:
            log.warn('MachineController created without a serial port')

    def block(self):
        # M400 - Wait for the queue to be empty before answering "OK"
        self._command('M400')

    def check_pause(self):
        while True:
            s = status.read()
            if s is not None and 'state' in s and s['state'] == 'paused':
                time.sleep(0.25)
                continue
            else:
                break

    def check_lunch_break(self, sleep_s=60.0):
        s = status.read()
        start_t = time.time()

        if s is not None and 'speed' in s and s['speed'] == 'slow':
            while time.time() - start_t < sleep_s:
                self.light(True)
                time.sleep(6.0)
                self.light(False)
                time.sleep(6.0)

    def reset(self):
        # send newlines to clear noise
        if self.serial_port:
            with self.serial_mutex:
                self.serial_port.flushInput()
                self.serial_port.write(NEWLINE)
                self.serial_port.write(NEWLINE)
                self.serial_port.write(NEWLINE)
        self.reset_emergency()
        self.reset_emergency()
        self.reset_emergency()

        self.motors(False)
        self.motors(True)
        self._command('G92 E0')   # Assume the head is set

    def home(self):
        self._command('G28')
        self.block()
        status.write(posx=0, posy=0, posz=0)

    def home_z(self):
        self._command('G28 Z0')
        self.block()
        status.write(posz=0)

    def home_e(self):
        raise Exception('home_e is not intended for the new machine')

        # big steps
        while True:
            self._command('G92 E0')   # reset E axis to 0
            result = self._command('M119', read_result=True)
            if result.find('min_z:1') > -1:
                break
            self.go(e=-3)
            self.block()
        # opposite side
        while True:
            self._command('G92 E0')   # reset E axis to 0
            result = self._command('M119', read_result=True)
            if result.find('min_z:0') > -1:
                self._command('G92 E0')   # reset E axis to 0
                break
            self.go(e=0.5)
            self.block()

        self.go(e=3.0)
        self._command('G92 E0')   # reset E axis to 0
        status.write(pose=0)

    def get_pickup_z(self):
        return self.pickup_z

    def get_scan_z(self):
        return self.scan_z

    def _check_movement(self, **kwargs):
        if 'x' in kwargs and kwargs['x'] is not None:
            if kwargs['x'] < self.limits_x[0] or kwargs['x'] > self.limits_x[1]:
                log.warn('Invalid movement: X=%f', kwargs['x'])
                return False
        if 'y' in kwargs and kwargs['y'] is not None:
            if kwargs['y'] < self.limits_y[0] or kwargs['y'] > self.limits_y[1]:
                log.warn('Invalid movement: Y=%f', kwargs['y'])
                return False
        if 'z' in kwargs and kwargs['z'] is not None:
            if kwargs['z'] < self.limits_z[0] or kwargs['z'] > self.limits_z[1]:
                log.warn('Invalid movement: Z=%f', kwargs['z'])
                return False
        if 'e' in kwargs and kwargs['e'] is not None:
            if kwargs['e'] < self.limits_e[0] or kwargs['e'] > self.limits_e[1]:
                log.warn('Invalid movement: E=%f', kwargs['e'])
                return False
        return True

    def check_movement(self, x=None, y=None, z=None, e=None):
        return self._check_movement(x=x, y=y, z=z, e=e)

    def feedrate(self, f):
        cmd_str = format_feed(f=f)
        self._command(cmd_str)

    def rapid(self, x=None, y=None, z=None, e=None):
        if self._check_movement(x=x, y=y, z=z, e=e):
            cmd_str = 'G0' + ' ' + format_pos(x=x, y=y, z=z, e=e)
            self._command(cmd_str)
            status.write(posx=x, posy=y, posz=z, pose=e)
            return True
        else:
            return False

    def go(self, x=None, y=None, z=None, e=None, force=False):
        if self._check_movement(x=x, y=y, z=z, e=e) or force:
            cmd_str = 'G1' + ' ' + format_pos(x=x, y=y, z=z, e=e)
            self._command(cmd_str)
            status.write(posx=x, posy=y, posz=z, pose=e)
            return True
        else:
            return False

    def dwell(self, ms):
        cmd_str = 'G4 P%d' % ms
        self._command(cmd_str)

    def scan_top(self, offset=0):
        """ Goes to the scan positio
            If offset is specified, it will stay the offset away from the scan position
        """
        z = min(self.scan_z, max(0.0, self.scan_z - offset))
        self.go(z=z)
        self.dwell(200)   
        # self.home_z()   # Also home for safety # TODO: really home?

    def pickup_top(self, offset=0):
        """ Goes to the top position
            If offset is specified, it will stay the offset away from the top position
        """
        z = min(self.pickup_z, max(0.0, self.pickup_z - offset))
        self.go(z=z)
        self.dwell(200)   
        # self.home_z()   # Also home for safety # TODO: really home?

    def pickup_g30(self):
        self.pickup_top()
        self.set_pickup_params(max_z=self.pickup_z)
        cmd_str = 'G30'
        result = self._command(cmd_str, read_result=True)
        if result.startswith('Z:'):  # Z:20.0270 C:445 - parse as z_delta
            m = re.search('Z:(\d+\.\d+)', result)
            if len(m.groups()) >= 1:
                z_delta = float(m.group(1))
                z_val = self.pickup_z - z_delta
                return z_val
        # result is "ZProbe not triggered" or something else
        # reset Z and switch off the vacuum; return nothing
        self.vacuum(False)
        self.home_z()
        return None

    def pickup_custom(self, start_z=24.0, end_z=0.0, step=1.5):   # end at 0.0 for the new sucker, 1.5 for the new
        self.pickup_top()

        has_picked = False
        pick_z = start_z
        self.eject(False)

        self.go(z=pick_z)
        self.vacuum(True)

        self.block()

        while pick_z > end_z and not has_picked:
            pick_z = max(end_z, pick_z - step)

            self.go(z=pick_z)
            self.block()

            if pick_z < end_z + 1.0:
                self.dwell(1500)
                self.block()

            result = self._command('M119', read_result=True)

            if result.find('Probe: 1') > -1:
                has_picked = True

        if has_picked:
            self.dwell(1000)
            self.block()
            self.pickup_top()

            result = self._command('M119', read_result=True)   # Read again
            if result.find('Probe: 1') > -1:
                return pick_z

        # reset Z and switch off the vacuum; return nothing
        self.vacuum(False)
        self.pickup_top(offset=9.0)   # Come near the homing, so it will be faster...
        
        self.pickup_top()
        # self.home_z()

        return None

    def set_pickup_params(self, slow_feed=None, fast_feed=None, return_feed=None, max_z=None, probe_height=None):
        max_z = max_z / 2  # weirdness, need to divide by 2
        cmd_str = 'M670'
        cmd_feed = format_feed(s=slow_feed, k=fast_feed, r=return_feed)
        cmd_pos = format_pos(z=max_z, h=probe_height)
        if cmd_feed:
            cmd_str += ' ' + cmd_feed
        if cmd_pos:
            cmd_str += ' ' + cmd_pos
        self._command(cmd_str)

    def motors(self, state):
        cmd_str = 'M17' if state else 'M18'
        self._command(cmd_str)

    def vacuum(self, state):
        cmd_str = 'M42' if state else 'M43'
        self._command(cmd_str)
        status.write(vacuum=True if state else False)

    def eject(self, state):
        cmd_str = 'M44' if state else 'M45'
        self._command(cmd_str)
        status.write(eject=True if state else False)

    def light(self, state, channel=None):
        if channel is not None:
            pwm_val = min(95, channel * 15 + 15)
        else:
            pwm_val = 98  # Almost 100%, but still a carrier...

        if state:
            cmd_str = 'M108 S%d' % (pwm_val, )
        else:
            cmd_str = 'M109'

        self._command(cmd_str)
        status.write(light=True if state else False)

    def emergency(self):
        cmd_str = 'M112'
        self._command(cmd_str)

    def reset_emergency(self):
        cmd_str = 'M999'
        self._command(cmd_str)

    def raw(self, cmd_str):
        self._command(cmd_str)

    def _command(self, cmd_str, read_result=False):
        log.debug('Sending command "%s"', cmd_str)
        if self.serial_port:
            with self.serial_mutex:
                self.serial_port.flushInput()
                self.serial_port.write(bytes(cmd_str, 'utf-8') + NEWLINE)
                # self.serial_port.flushOutput()
                line = self.serial_port.readline().decode('utf-8')
                log.debug('Received line "%s"', line.rstrip())
                if read_result:
                    if line.startswith('ok'):
                        raise StateException('Expected result, but OK returned')
                    result = line
                    line = self.serial_port.readline().decode('utf-8')
                    log.debug('Received line#2 "%s"', line.rstrip())
                else:
                    result = None
                if line.lower().startswith('ok'):
                    return result
                elif line.lower().startswith('!!'):
                    raise StateException('Emergency')
                else:
                    raise CommunicationException('Communication Error')

    def _close(self):
        if self.serial_port:
            with self.serial_mutex:
                self.serial_port.flushInput()
                self.serial_port.flushOutput()
                self.serial_port.close()
                self.serial_port = None

    def __del__(self):
        self._close()


class StateException(Exception):
    pass


class CommunicationException(Exception):
    pass


class SerialOpenError(Exception):
    def __init__(self, port, baud):
        Exception.__init__(self)
        self.message = 'Cannot open port %s at %d bps' % (port, baud)
        self.port = port
        self.baud = baud

    def __str__(self):
        return self.message


if __name__ == '__main__':
    try: # raspi detection
        host = netifaces.ifaddresses(IFACE)[netifaces.AF_INET][0]['addr']
    except:
        host = 'localhost'
    daemon = Pyro4.Daemon(host=host, port=PORT)
    control = MachineController(DEVICE)
    uri = daemon.register(control, 'control')
    log.info('Running at %s', uri)
    daemon.requestLoop()
