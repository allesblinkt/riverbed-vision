#!/usr/bin/python
import Pyro4
import time
import sys

hostname = '192.168.0.27'

uri = 'PYRO:control@{}:5001'.format(hostname)
control = Pyro4.Proxy(uri)

def lift_up():
	control.vacuum(True)
	time.sleep(1.0)
	h = control.pickup()
	assert h
	return h

def lift_down(h):
	control.go(z=max(h - 3, 0))
	control.vacuum(False)
	time.sleep(0.1)
	control.pickup_top()

def move_stone(x1, y1, e1, x2, y2, e2):
	control.go(e=e1)
	control.go(x=x1, y=y1)
	h = lift_up()
	control.go(e=e2)
	control.go(x=x2, y=y2)
	lift_down(h)

while True:
	move_stone(0, 0, 0, 500, 250, 90)
	move_stone(500, 250, 90, 0, 0, 0)
