from flask import Flask, render_template, jsonify, request, abort
import flask_shelve as shelve

import trainer
import random


app = Flask(__name__)
app.config['SHELVE_FILENAME'] = 'voting.db'
shelve.init_app(app)


stones = trainer.load_stones()


def parse_int_default(s, default=0):
    try:
        return int(s)
    except (ValueError, TypeError):
        return default


def parse_float_default(s, default=0.0):
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def abort_if_stone_doesnt_exist(stone_id):
    if stone_id not in stones:
        abort(404, message="Stone {} doesn't exist".format(stone_id))


def abort_if_category_doesnt_exist(category_id, db):
    if category_id not in db["category_stones"]:
        abort(404, message="Category {} doesn't exist".format(category_id))


def stone_to_dict(stone, db):
    def rgbtriplet2hex(triplet):
        return '#%02x%02x%02x' % (triplet[0], triplet[1], triplet[2])

    id = stone.identifier
    votes = db["stone_category_votes"][id] if id in db["stone_category_votes"] else []

    return {
        'identifier': stone.identifier,
        'img_src': '/static/stones/' + stone.identifier + '.png',
        'brightness': stone.color.tolist()[0],
        'color': rgbtriplet2hex(stone.color.tolist()),
        'structure': stone.structure.tolist(),
        'votes': votes
    }


# @app.before_first_request
# def initialize():
#     db = shelve.get_shelve('c')

#     print(len(db["stone_category_votes"]))
#     db.close()

#     return

#     print("Creating new")
#     db = shelve.get_shelve('c')
#     db["stone_category_votes"] = {
#         "stone_grab_3432_0690.jpg_008": ["a", "a"],
#         "stone_grab_1092_0345.jpg_009": ["b"],
#         "stone_grab_0312_0345.jpg_005": ["c"],
#         "stone_grab_1716_0621.jpg_005": ["d"],
#         "stone_grab_1365_1035.jpg_005": ["e", "e", "e", "e", "e", "e"]
#     }

#     db["stone_vote_sequences"] = {
        
#     }

#     db["category_stones"] = {  # Rebuild ...
#         "a": set(["stone_grab_3432_0690.jpg_008"]),
#         "b": set(["stone_grab_1092_0345.jpg_009"]),
#         "c": set(["stone_grab_0312_0345.jpg_005"]),
#         "d": set(["stone_grab_1716_0621.jpg_005"]),
#         "e": set(["stone_grab_1365_1035.jpg_005"])
#     }

#     db["stone_vote_indexer"] = 0
#     db["category_indexer"] = 4  # Rebuild

#     db.close()


@app.route('/picker/')
def page_picker(name=None):
    return render_template('picker.html', name='Yello')


@app.route('/stones/category/<string:category_id>')
def stones_list_category(category_id):
    db = shelve.get_shelve('r')

    abort_if_category_doesnt_exist(category_id, db)

    cd = category_dict(category_id, db, stone_count=None)
    return jsonify({"stones": cd["highest_voted"]})


@app.route('/stones/random')
def stones_list_random():
    db = shelve.get_shelve('r')

    count = parse_int_default(request.values.get('count'), 10)
    l = list(stones.values())
    random.shuffle(l)
    l = l[:count]

    r = [stone_to_dict(stone, db) for stone in l]

    return jsonify({'stones': r})


@app.route('/stones/similar/<string:stone_id>')
def stones_list_similar(stone_id):
    db = shelve.get_shelve('r')

    abort_if_stone_doesnt_exist(stone_id)

    reference_stone = stones[stone_id]

    stone_category_votes = db["stone_category_votes"]
    stone_vote_sequences = db["stone_vote_sequences"]

    stones_with_votecount = []
    for stone_id in stones:
        votes = 0
        vote_prog = 0

        if stone_id in stone_category_votes:
            votes = len(stone_category_votes[stone_id])

        if stone_id in stone_vote_sequences:
            vote_prog = stone_vote_sequences[stone_id]

        stones_with_votecount.append((stones[stone_id], votes, vote_prog))

    min_votes = min(stones_with_votecount, key=lambda tup: tup[1])[1]
    max_votes = max(stones_with_votecount, key=lambda tup: tup[1])[1]

    votes_thresh = (max_votes - min_votes) // 2

    min_vote_prog = min(stones_with_votecount, key=lambda tup: tup[2])[2]
    max_vote_prog = max(stones_with_votecount, key=lambda tup: tup[2])[2]
    vote_prog_thresh = (max_vote_prog - min_vote_prog) // 10

    selection = [s[0] for s in stones_with_votecount if s[1] <= votes_thresh and s[2] <= vote_prog_thresh]

    # Parameters
    count = parse_int_default(request.values.get('count'), 10)
    # count = 10

    find_count = count * 5
    l = trainer.find_best_matches(reference_stone, selection, find_count)
    random.shuffle(l)
    l = l[:count]

    r = [stone_to_dict(stone, db) for stone in l]

    return jsonify({'stones': r})


def category_dict(category_id, db, stone_count=10):
    stone_ids = list(db["category_stones"][category_id])
    stone_category_votes = db["stone_category_votes"]

    stones_with_votecount = []
    for stone_id in stone_ids:
        if stone_id in stone_category_votes:
            stone = stones[stone_id]
            votes = stone_category_votes[stone_id].count(category_id)

            stones_with_votecount.append((stone, votes))
        else:
            assert("stone_id should exist in stone_category_votes")

    stones_with_votecount = sorted(stones_with_votecount, key=lambda tup: tup[1], reverse=True)

    if stone_count:
        # stones_with_votecount = stones_with_votecount[:stone_count * 10]
        # random.shuffle(stones_with_votecount)
        stones_with_votecount = stones_with_votecount[:stone_count]

    highest = [stone_to_dict(s[0], db) for s in stones_with_votecount]

    brightness = 0

    if len(highest) > 0:
        brightness = highest[0]["brightness"]

    return {'id': category_id, 'stone_ids': stone_ids, 'highest_voted': highest, 'brightness': brightness}


@app.route("/categories")
def categories_list():
    db = shelve.get_shelve('r')

    category_stones = db["category_stones"].keys()

    l = []
    for category_id in category_stones:
        l.append(category_dict(category_id, db))

    l = sorted(l, key=lambda c: c["brightness"], reverse=True)

    return jsonify({"categories": l})


@app.route("/stonevotes/count")
def stonevotes_count():
    db = shelve.get_shelve('r')
    count = len(db["stone_category_votes"])

    return jsonify({"count": count})


@app.route('/category/<string:category_id>')
def category_get(category_id):
    db = shelve.get_shelve('r')

    abort_if_category_doesnt_exist(category_id, db)

    cd = category_dict(category_id, db, stone_count=None)

    return jsonify({stones: cd["highest_voted"]})


@app.route('/category/add')
def category_add():
    db = shelve.get_shelve('c')

    db["category_indexer"] += 1
    category_id = str(db["category_indexer"])

    category_stones = db["category_stones"]
    category_stones[category_id] = set()
    db["category_stones"] = category_stones
    return jsonify({'created': True, 'category_id': category_id})


@app.route('/category/merge/<string:category_id_a>/<string:category_id_b>')
def category_merge(category_id_a, category_id_b):
    """Merge a into b"""

    db = shelve.get_shelve('c')

    abort_if_category_doesnt_exist(category_id_a, db)
    abort_if_category_doesnt_exist(category_id_b, db)

    stone_category_votes = db["stone_category_votes"]

    for stone_id in stone_category_votes:
        print(stone_id)
        old_votes = stone_category_votes[stone_id]
        print(old_votes)

        new_votes = [category_id_b if v == category_id_a else v for v in old_votes]
        stone_category_votes[stone_id] = new_votes

    category_stones = db["category_stones"]
    category_stones[category_id_b].update(category_stones[category_id_a])
    del category_stones[category_id_a]

    db["category_stones"] = category_stones
    db["stone_category_votes"] = stone_category_votes

    return jsonify({'merged': True})


@app.route('/stones/<string:stone_id>/vote/<string:category_id>')
def stone_vote_category(stone_id, category_id):
    db = shelve.get_shelve('c')

    abort_if_stone_doesnt_exist(stone_id)
    abort_if_category_doesnt_exist(category_id, db)

    db["stone_vote_indexer"] += 1

    stone_vote_sequences = db["stone_vote_sequences"]
    stone_vote_sequences[stone_id] = db["stone_vote_indexer"]
    db["stone_vote_sequences"] = stone_vote_sequences

    if stone_id not in db["stone_category_votes"]:
        stone_category_votes = db["stone_category_votes"]
        stone_category_votes[stone_id] = []
        db["stone_category_votes"] = stone_category_votes

    stone_category_votes = db["stone_category_votes"]
    stone_category_votes[stone_id].append(category_id)

    category_stones = db["category_stones"]
    category_stones[category_id].add(stone_id)

    db["stone_category_votes"] = stone_category_votes
    db["category_stones"] = category_stones

    return jsonify({'counted': True})


@app.route('/stones/<string:stone_id>/unvote/<string:category_id>')
def stone_unvote_category(stone_id, category_id):
    db = shelve.get_shelve('c')

    abort_if_stone_doesnt_exist(stone_id)
    abort_if_category_doesnt_exist(category_id, db)

    stone_category_votes = db["stone_category_votes"]
    category_stones = db["category_stones"]

    old_votes = stone_category_votes[stone_id]
    new_votes = [v for v in old_votes if v != category_id]
    stone_category_votes[stone_id] = new_votes

    if stone_id in category_stones[category_id]:
        category_stones[category_id].remove(stone_id)

        stone_vote_sequences = db["stone_vote_sequences"]
        stone_vote_sequences[stone_id] = 0
        db["stone_vote_sequences"] = stone_vote_sequences

    db["stone_category_votes"] = stone_category_votes
    db["category_stones"] = category_stones

    return jsonify({'uncounted': True})


# @user.route('/<user_id>', defaults={'username': None})
# @user.route('/<user_id>/<username>')
# def show(user_id, username):


@app.route('/stones/<string:stone_id>/votes')
def stone_votes_list(stone_id):
    db = shelve.get_shelve('r')

    abort_if_stone_doesnt_exist(stone_id)

    votes = []
    if stone_id in db["stone_category_votes"]:
        votes = db["stone_category_votes"][stone_id]

    return jsonify({'votes': votes})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)

