#!/usr/bin/python
import re
import serial
import threading
import logging
import Pyro4

NEWLINE = '\r\n'
DEVICE  = None # '/dev/ttyACM0'
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

        if port_name:
            log.info('Opening serial port %s at %d bps', port_name, baudrate)
            try:
                self.serial_port = serial.Serial(port_name)
                self.serial_port.setTimeout(1.0)

                self.serial_port.baudrate = baudrate

                self.reading_thread = threading.Thread(target=self._threaded_read)
                self.reading_thread.daemon = True
                self.reading_thread.start()
            except SerialOpenError:
                raise SerialOpenError(port_name, baudrate)
        else:
            log.warn('MachineController created without a serial port')

    def home(self):
        self._command('$X')
        self._command('$H')

    def check_movement(self, **kwargs):
        # CHECK bounds
        # CHECK if rapid in allowed rapid height
        return True

    def rapid(self, x=None, y=None, z=None):
        if self.check_movement(x=x, y=y, z=z):
            cmd_str = 'G00' + ' ' + format_pos(x=x, y=y, z=z)
            self._command(cmd_str)

    def go(self, x=None, y=None, z=None):
        if self.check_movement(x=x, y=y, z=z):
            cmd_str = 'G01' + ' ' + format_pos(x=x, y=y, z=z)
            self._command(cmd_str)

    def pickup(self):
        cmd_str = 'G30'
        self._command(cmd_str)

    def set_pickup_params(self, slow_feed=None, fast_feed=None, return_feed=None, max_z=None, probe_height=None):
        cmd_str = 'M670'
        cmd_feed = format_feed(s=slow_feed, k=fast_feed, r=return_feed)
        cmd_pos = format_pos(z=max_z, h=probe_height)
        if cmd_feed:
            cmd_str += ' ' + cmd_feed
        if cmd_pos:
            cmd_str += ' ' + cmd_pos
        self._command(cmd_str)

    def save_pickup_params(self):
        cmd_str = 'M500'
        self._command(cmd_str)

    def read_vacuum(self):
        cmd_str = 'M119'
        self._command(cmd_str)

        read = self._response()

        rs = 'Probe: (\d+)'

        m = re.search(rs, read)

        if len(m.groups()) and m.group(1) == '255':
            # True
            pass
        else:
            # False
            pass

    def switch_vacuum(self, state):
        cmd_str = 'M106' if state else 'M0107'
        self._command(cmd_str)

    def switch_throwoff(self, state):
        cmd_str = 'M108' if state else 'M0109'
        self._command(cmd_str)

    def close(self):
        if self.serial_port:
            with self.serial_mutex:
                self.serial_port.flushInput()
                self.serial_port.flushOutput()
                self.serial_port.close()
                self.serial_port = None

    def _command(self, cmd_str):
        log.debug('Sending command "%s"', cmd_str)
        if self.serial_port:
            with self.serial_mutex:
                self.serial_port.write(cmd_str + NEWLINE)
                self.serial_port.flushOutput()

    def _response(self):
        try:
            return self.serial_port.readline()
        except:
            return ''

    def __del__(self):
        self.close()


class SerialOpenError(Exception):
    def __init__(self, port, baud):
        Exception.__init__(self)
        self.message = 'Cannot open port %s at %d bps' % (port, baud)
        self.port = port
        self.baud = baud

    def __str__(self):
        return self.message


if __name__ == '__main__':
    daemon = Pyro4.Daemon(port=PORT)
    control = MachineController(DEVICE)
    uri = daemon.register(control, 'control')
    print 'Running at', uri
    daemon.requestLoop()
