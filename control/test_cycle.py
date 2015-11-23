#!/usr/bin/python
import Pyro4
import time
import sys

hostname = '192.168.0.27'

uri = 'PYRO:control@{}:5001'.format(hostname)
control = Pyro4.Proxy(uri)

def dodo(x1, y1, e1, x2, y2, e2):
	control.go(e=e1)
	control.go(x=x1, y=y1)
	control.vacuum(True)
	time.sleep(1.0)
	control.go(z=38)
	h = control.pickup()
	assert h
	control.go(e=e2)
	control.go(x=x2, y=y2)
	h = max(h - 3, 0)
	control.go(z=h)
	control.vacuum(False)
	time.sleep(0.1)
	control.go(z=38)

while True:
	dodo(0, 0, 0, 500, 250, 90)
	dodo(500, 250, 90, 0, 0, 0)
