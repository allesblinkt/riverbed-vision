#!/usr/bin/ipython -i
import Pyro4

uri = 'PYRO:control@localhost:5001'
control = Pyro4.Proxy(uri)
