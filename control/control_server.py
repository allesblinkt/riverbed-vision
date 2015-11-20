#!/usr/bin/python
import re
import serial
import threading
import logging
import Pyro4

log = logging.getLogger(__name__)


def format_pos(pos):
    return '{:0.2f}'.format(pos)


def format_feed(feed):
    return '{:0.1f}'.format(feed)


def pos_string(x=None, y=None, z=None):
    x_str = format_pos(x)
    y_str = format_pos(y)
    z_str = format_pos(z)

    return 'X{} Y{} Z{}'.format(x_str, y_str, z_str)


class MachineController(object):
    """MachineController"""

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
            log.warn('Hardware created without a serial port')

    def check_movement(self, **kwargs):
        # CHECK bounds
        # CHECK if rapid in allowed rapid height

        pass

    def rapid(self, x=None, y=None, z=None):
        self.check_movement(x, y, z)

        cmd_str = 'G00' + ' ' + pos_string(x, y, z)
        self._command(cmd_str)

    def go(self, x=None, y=None, z=None):
        self.check_movement(x, y, z)

        cmd_str = 'G01' + ' ' + pos_string(x, y, z)
        self._command(cmd_str)

    def pickup(self):
        cmd_str = 'G30'

        self._command(cmd_str)

    def set_pickup_params(self, slow_feed=None, fast_feed=None, return_feed=None, max_z=None, probe_height=None):
        cmd_str = 'M670'

        if slow_feed:
            cmd_str += ' S' + format_feed(slow_feed)
        if fast_feed:
            cmd_str += ' K' + format_feed(fast_feed)
        if return_feed:
            cmd_str += ' R' + format_feed(return_feed)
        if max_z:
            cmd_str += ' Z' + format_pos(max_z)
        if probe_height:
            cmd_str += ' H' + format_pos(probe_height)

        self._command(cmd_str)

    def save_pickup_params(self):
        cmd_str = 'M500'

        self._command(cmd_str)

    def read_vacuum(self):
        cmd_str = 'M119'
        self._command(cmd_str)

        read = self._read_back()

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
        """ Be nice, close the serial port. """
        if self.serial_port:
            with self.serial_mutex:
                self.serial_port.flushInput()
                self.serial_port.flushOutput()
                self.serial_port.close()
                self.serial_port = None

    def _command(self, cmd_str):
        log.debug('Sending command "%s"', cmd_str.rstrip())

        if self.serial_port:
            with self.serial_mutex:
                self.serial_port.write(cmd_str + self.CMD_NEWLINE)
                self.serial_port.flushOutput()

    def _read_back(self):
        self.serial_port.readline()   # TODO: read and handle error
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
    daemon = Pyro4.Daemon(port=5001)
    uri = daemon.register(MachineController, 'control')
    print 'Running at', uri
    daemon.requestLoop()
