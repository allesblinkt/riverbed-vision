#!/usr/bin/ipython -i
import Pyro4
import sys

if len(sys.argv) > 1:
    hostname = sys.argv[1]
else:
    hostname = 'localhost'

uri = 'PYRO:control@{}:5001'.format(hostname)
control = Pyro4.Proxy(uri)
