from flask import Flask, render_template, jsonify, request, abort

import trainer
import random

app = Flask(__name__)

stones = trainer.load_stones()
stone_category_votes = {}
category_stones = {
    "a": set()
}


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


def abort_if_category_doesnt_exist(category_id):
    if category_id not in category_stones:
        abort(404, message="Category {} doesn't exist".format(category_id))


def stone_to_dict(stone):
    def rgbtriplet2hex(triplet):
        return '#%02x%02x%02x' % (triplet[0], triplet[1], triplet[2])

    id = stone.identifier
    votes = stone_category_votes[id] if id in stone_category_votes else []

    return {
        'identifier': stone.identifier,
        'img_src': '/static/stones/' + stone.identifier + '.png',
        'color': rgbtriplet2hex(stone.color.tolist()),
        'structure': stone.structure.tolist(),
        'votes': votes
    }


@app.route('/picker/')
def page_picker(name=None):
    return render_template('picker.html', name='Yello')


@app.route('/stones/random')
def stones_list_random():
    count = 10
    l = stones.values()
    random.shuffle(l)

    r = [stone_to_dict(stone) for stone in l]

    return jsonify({'stones': r[:count]})


@app.route('/stones/similar/<string:stone_id>')
def stones_list_similar(stone_id):
    abort_if_stone_doesnt_exist(stone_id)

    # Parameters
    count = parse_int_default(request.values.get('count'), 10)
    # count = 10
    l = stones.values()

    stone = stones[stone_id]

    l = trainer.find_best_matches(stone, l)
    r = [stone_to_dict(stone) for stone in l]

    return jsonify({'stones': r[:count]})


@app.route('/stones/<string:stone_id>/vote/<string:category_id>')
def stone_vote_category(stone_id, category_id):
    abort_if_stone_doesnt_exist(stone_id)
    abort_if_category_doesnt_exist(category_id)

    if stone_id not in stone_category_votes:
        stone_category_votes[stone_id] = []

    stone_category_votes[stone_id].append(category_id)
    category_stones[category_id].add(stone_id)

    print(stone_category_votes)
    print(category_stones)

    return jsonify({'counted': True})



@app.route('/category/<string:category_id>')
def category_get(category_id):
    abort_if_category_doesnt_exist(category_id)

    stone_ids = list(category_stones[category_id])

    stones_with_votecount = []
    for stone_id in stone_ids:
        if stone_id in stone_category_votes:
            stone = stones[stone_id]
            votes = stone_category_votes[stone_id].count(category_id)

            stones_with_votecount.append((stone, votes))
            print()
        else:
            assert("SSS")

    stones_with_votecount = sorted(stones_with_votecount, key=lambda tup: tup[1], reverse=True)

    highest = [stone_to_dict(s[0]) for s in stones_with_votecount][:10]
    return jsonify({'id': category_id, 'stone_ids': stone_ids, 'highest_voted': highest})


@app.route('/category/add/<string:category_id>')
def category_add(category_id):
    if category_id in category_stones:
        return jsonify({'created': False})
    else:    
        category_stones[category_id] = set()
        return jsonify({'created': True})



@app.route('/stones/<string:stone_id>/unvote/<string:category_id>')
def stone_unvote_category(stone_id, category_id):
    abort_if_stone_doesnt_exist(stone_id)
    abort_if_category_doesnt_exist(category_id)

    if stone_id in stone_category_votes and category_id in stone_category_votes[stone_id]:
        stone_category_votes[stone_id].remove(category_id)   # Removes one...

        if category_id not in stone_category_votes[stone_id]:
            category_stones[category_id].remove(stone_id)

    return jsonify({'uncounted': True})


# @user.route('/<user_id>', defaults={'username': None})
# @user.route('/<user_id>/<username>')
# def show(user_id, username):


@app.route('/stones/<string:stone_id>/votes')
def stone_votes_list(stone_id):
    abort_if_stone_doesnt_exist(stone_id)

    votes = []
    if stone_id in stone_category_votes:
        votes = stone_category_votes[stone_id]

    return jsonify({'votes': votes})


if __name__ == '__main__':
    app.run(debug=True)
