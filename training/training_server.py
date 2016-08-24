from flask import Flask, render_template, jsonify
from flask_restful import reqparse, abort, Api, Resource

import trainer
import random

app = Flask(__name__)
api = Api(app)


stones = {
    'stone1': {'img_src': 'foo.jpg', 'task': 'build an API'},
    'stone2': {'img_src': 'foo.jpg', 'task': '?????'},
    'stone3': {'img_src': 'foo.jpg', 'task': 'profit!'},
}


stones = trainer.load_stones()


def abort_if_stone_doesnt_exist(todo_id):
    if stone_id not in stones:
        abort(404, message="Stone {} doesn't exist".format(stone_id))

parser = reqparse.RequestParser()
parser.add_argument('task')



@app.route('/picker/')
#@app.route('/hello/<name>')
def page_picker(name=None):
    return render_template('picker.html', name='Yello')


# Todo
# shows a single todo item and lets you delete a todo item
class ApiStone(Resource):
    def get(self, stone_id):
        abort_if_todo_doesnt_exist(todo_id)
        return stone[todo_id]

    # def put(self, todo_id):
    #     args = parser.parse_args()
    #     task = {'task': args['task']}
    #     TODOS[todo_id] = task
    #     return task, 201


# TodoList
# shows a list of all todos, and lets you POST to add new tasks
class ApiStoneList(Resource):
    def get(self):
        return stones

    def post(self):
        args = parser.parse_args()
        stone_id = int(max(TODOS.keys()).lstrip('stone')) + 1
        stone_id = 'stone%i' % stone_id
        stones[stone_id] = {'task': args['task']}
        return stones[stone_id], 201



def stone_to_dict(stone):
    def rgbtriplet2hex(triplet):
        return '#%02x%02x%02x' % (triplet[0], triplet[1], triplet[2])


    return {
        'identifier': stone.identifier,
        'img_src': '/static/stones/' + stone.identifier + '.png',
        'color': rgbtriplet2hex(stone.color.tolist()),
        'structure': stone.structure.tolist(),

    }

@app.route('/stones/random')
def stones_random_list():
    count = 10
    l = stones.copy()
    random.shuffle(l)

    r = [stone_to_dict(stone) for stone in l]

    return jsonify({'stones': r[:count]})

@app.route('/stones/similar')
def stones_similar_list():
    count = 10
    l = stones.copy()
    random.shuffle(l)

    l = trainer.find_best_matches(stones[12], l)

    r = [stone_to_dict(stone) for stone in l]

    return jsonify({'stones': r[:count]})




##
## Actually setup the Api resource routing here
##
#api.add_resource(ApiStoneQuery, '/stones/similar')
api.add_resource(ApiStoneList, '/stones')
api.add_resource(ApiStone, '/stones/<todo_id>')


if __name__ == '__main__':
    app.run(debug=True)