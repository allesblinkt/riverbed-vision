import shelve

def read(keys=[]):
    try:
        d = shelve.open('/tmp/jller.status', 'r')
        v = {}
        for k in keys:
            v[k] = d[k]
        d.close()
        return v
    except:
        return None

def write(**data):
    for _ in range(10): # try writing 10 times
        try:
            d = shelve.open('/tmp/jller.status', 'c')
            for k in data.keys():
                d[k] = data[k]
            d.close()
            return
        except OSError as e:
            pass
