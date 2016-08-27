from flask import Flask, render_template, jsonify
from flask_restful import reqparse, abort, Api, Resource

import trainer
import random

app = Flask(__name__)
api = Api(app)

stones = trainer.load_stones()
stone_category_votes = {}
categories = {}


def abort_if_stone_doesnt_exist(stone_id):
    if stone_id not in stones:
        abort(404, message="Stone {} doesn't exist".format(stone_id))


def abort_if_category_doesnt_exist(category_id):
    if category_id not in categories:
        abort(404, message="Category {} doesn't exist".format(category_id))


def stone_to_dict(stone):
    def rgbtriplet2hex(triplet):
        return '#%02x%02x%02x' % (triplet[0], triplet[1], triplet[2])

    return {
        'identifier': stone.identifier,
        'img_src': '/static/stones/' + stone.identifier + '.png',
        'color': rgbtriplet2hex(stone.color.tolist()),
        'structure': stone.structure.tolist(),
    }



parser = reqparse.RequestParser()
parser.add_argument('task')



@app.route('/picker/')
#@app.route('/hello/<name>')
def page_picker(name=None):
    return render_template('picker.html', name='Yello')



@app.route('/stones/random')
def stones_list_random():
    count = 10
    l = stones.copy()
    random.shuffle(l)

    r = [stone_to_dict(stone) for stone in l]

    return jsonify({'stones': r[:count]})


@app.route('/stones/similar')
def stones_list_similar():
    count = 10
    l = stones.copy()
    random.shuffle(l)

    l = trainer.find_best_matches(stones[12], l)

    r = [stone_to_dict(stone) for stone in l]

    return jsonify({'stones': r[:count]})


@app.route('/stones/<str:stone_id>/vote/<str:category_id>')
def stone_vote_category(stone_id, category_id):
    abort_if_stone_doesnt_exist(stone_id)
    abort_if_category_doesnt_exist(category_id)

    if stone_id not in stone_category_votes:
        stone_category_votes[stone_id] = []

    stone_category_votes.append(category_id)
    categories[category_id].add(stone_id)

    return jsonify({'counted': True})


@app.route('/stones/<str:stone_id>/unvote/<str:category_id>')
def stone_unvote_category(stone_id, category_id):
    abort_if_stone_doesnt_exist(stone_id)
    abort_if_category_doesnt_exist(category_id)

    if stone_id in stone_category_votes and category_id in stone_category_votes[stone_id]:
        stone_category_votes[stone_id].remove(category_id)   # Removes one...

        if category_id not in stone_category_votes[stone_id]:
            categories[category_id]['stones_set'].remove(stone_id)

    return jsonify({'uncounted': True})


@app.route('/stones/<str:stone_id>/votes')
def stone_votes_list(stone_id):
    abort_if_stone_doesnt_exist(stone_id)

    votes = []
    if stone_id in stone_category_votes:
        votes = stone_category_votes[stone_id]

    return jsonify({'votes': votes})





if __name__ == '__main__':
    app.run(debug=True)