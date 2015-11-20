""" Use with ipython -i to get an interactive session """

import Pyro4

# use the URI that the server printed:
uri = "PYRO:api@localhost:9999"
api = Pyro4.Proxy(uri)