#!/usr/bin/ipython -i
import Pyro4
import time
import sys

if len(sys.argv) > 1:
    hostname = sys.argv[1]
else:
    hostname = 'localhost'

uri = 'PYRO:control@{}:5001'.format(hostname)
control = Pyro4.Proxy(uri)

control.home()
control.home_e()
control.block()
control.go(y=100, x=100)
control.feedrate(25000)
for i in range(210):
    control.go(x=100, y=100, e=0)
    control.block()
    control.go(x=3500, y=1700, e=180)
    control.block()
    control.go(e=0)
    control.block()
    control.go(e=180)
    control.block()

