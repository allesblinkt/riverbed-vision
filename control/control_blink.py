#!/usr/bin/env python3
import Pyro4
import time
import sys

if len(sys.argv) > 1:
    hostname = sys.argv[1]
else:
    hostname = 'localhost'

hostname = '10.0.42.42'

uri = 'PYRO:control@{}:5001'.format(hostname)
control = Pyro4.Proxy(uri)

l = 0

while True:
    control.light(True, l + 1)
    time.sleep(1)
    control.light(False, l + 1)
    l = (l + 1) % 3
