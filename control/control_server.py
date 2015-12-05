#!/usr/bin/python
import re
import serial
import threading
import logging
import Pyro4
import netifaces

NEWLINE = '\r\n'
DEVICE  = '/dev/ttyAMA0'
PORT    = 5001

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.DEBUG)
log = logging.getLogger(__name__)


def format_feed(**kwargs):
    return ' '.join(['{}{:0.1f}'.format(k.upper(), v) for k, v in sorted(kwargs.items()) if v is not None])


def format_pos(**kwargs):
    return ' '.join(['{}{:0.2f}'.format(k.upper(), v) for k, v in sorted(kwargs.items()) if v is not None])


class MachineController(object):

    def __init__(self, port_name, baudrate=115200):
        super(MachineController, self).__init__()

        self.serial_mutex = threading.Lock()
        self.serial_port = None
        self.port_name = port_name
        self.pickup_z = 38.0

        if port_name:
            log.info('Opening serial port %s at %d bps', port_name, baudrate)
            try:
                self.serial_port = serial.Serial(port_name)
                # self.serial_port.setTimeout(1.0)
                self.serial_port.baudrate = baudrate
            except SerialOpenError:
                raise SerialOpenError(port_name, baudrate)
        else:
            log.warn('MachineController created without a serial port')

    def block(self):
        # M400 - Wait for the queue to be empty before answering "OK"
        self._command('M400')

    def home(self, only_z=False):
        if only_z:
            self._command('G28 Z0')
        else:
            self._command('G28')
        self.block()

    def _check_movement(self, **kwargs):
        if 'x' in kwargs and kwargs['x'] is not None:
            if kwargs['x'] < 0 or kwargs['x'] > 1000:
                log.warn('Invalid movement: X=%f', kwargs['x'])
                return False
        if 'y' in kwargs and kwargs['y'] is not None:
            if kwargs['y'] < 0 or kwargs['y'] > 1730:
                log.warn('Invalid movement: Y=%f', kwargs['y'])
                return False
        if 'z' in kwargs and kwargs['z'] is not None:
            if kwargs['z'] < 0 or kwargs['z'] > 38:
                log.warn('Invalid movement: Z=%f', kwargs['z'])
                return False
        if 'e' in kwargs and kwargs['e'] is not None:
            if kwargs['e'] < 0 or kwargs['e'] > 180:
                log.warn('Invalid movement: E=%f', kwargs['e'])
                return False
        return True

    def feedrate(self, f):
        cmd_str = format_feed(f=f)
        self._command(cmd_str)

    def rapid(self, x=None, y=None, z=None, e=None):
        if self._check_movement(x=x, y=y, z=z, e=e):
            cmd_str = 'G0' + ' ' + format_pos(x=x, y=y, z=z, e=e)
            self._command(cmd_str)

    def go(self, x=None, y=None, z=None, e=None):
        if self._check_movement(x=x, y=y, z=z, e=e):
            cmd_str = 'G1' + ' ' + format_pos(x=x, y=y, z=z, e=e)
            self._command(cmd_str)

    def pickup_top(self):
        self.go(z=self.pickup_z)

    def pickup(self):
        self.pickup_top()
        self.set_pickup_params(max_z=self.pickup_z)
        cmd_str = 'G30'
        result = self._command(cmd_str, read_result=True)
        if result.startswith('Z:'): # Z:20.0270 C:445 - parse as z_delta
            m = re.search('Z:(\d+\.\d+)', result)
            if len(m.groups()) >= 1:
                z_delta = float(m.group(1))
                z_val = self.pickup_z - z_delta
                return z_val
        # result is "ZProbe not triggered" or something else
        # reset Z and switch off the vacuum; return nothing
        self.vacuum(False)
        self.home(only_z=True)
        return None

    def set_pickup_params(self, slow_feed=None, fast_feed=None, return_feed=None, max_z=None, probe_height=None):
        max_z = max_z / 2 # weirdness, need to divide by 2
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

    def light(self, state):
        cmd_str = 'M108' if state else 'M109'
        self._command(cmd_str)

    def _command(self, cmd_str, read_result=False):
        log.debug('Sending command "%s"', cmd_str)
        if self.serial_port:
            with self.serial_mutex:
                self.serial_port.write(cmd_str + NEWLINE)
                self.serial_port.flushInput()
                self.serial_port.flushOutput()
                line = self.serial_port.readline()
                log.debug('Received line "%s"', line.rstrip())
                if read_result:
                    if line.startswith('ok'):
                        raise StateException('Expected result, but OK returned')
                    result = line
                    line = self.serial_port.readline()
                    log.debug('Received line#2 "%s"', line.rstrip())
                else:
                    result = None
                if line.lower().startswith('ok'):
                    return result
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
        host = netifaces.ifaddresses('eth0')[netifaces.AF_INET][0]['addr']
    except:
        host = 'localhost'
    daemon = Pyro4.Daemon(host=host, port=PORT)
    control = MachineController(DEVICE)
    uri = daemon.register(control, 'control')
    print 'Running at', uri
    daemon.requestLoop()
