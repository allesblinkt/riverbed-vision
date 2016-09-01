import shelve

if __name__ == '__main__':
    old_db = shelve.open('voting.db')
    new_db = shelve.open('voting_shrinked.db')

    for k in old_db.keys():
        new_db[k] = old_db[k]

    new_db.close()
    old_db.close()
