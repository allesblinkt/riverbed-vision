import shelve

def read(keys=[]):
    try:
        d = shelve.open('/tmp/jller.status', 'r')
        v = dict(d)
        d.close()
        return v
    except:
        return {}

def write(**data):
    for _ in range(10): # try writing 10 times
        try:
            d = shelve.open('/tmp/jller.status', 'c')
            for k in data.keys():
                if data[k] is not None:
                    d[k] = data[k]
            d.close()
            return
        except OSError as e:
            pass
