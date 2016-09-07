#!/usr/bin/env python3

from flask import Flask, render_template
from flask.json import jsonify

import status

app = Flask(__name__)

@app.route('/')
def root():
    return render_template('page.html')

@app.route('/status.json')
def status_json():
    s = status.read(['state'])
    return jsonify(s)

@app.route('/sleep', methods=['POST'])
def sleep():
    status.write('state', 'sleeping')
    return 'ok'

@app.route('/unsleep', methods=['POST'])
def unsleep():
    status.write('state', 'working')
    return 'ok'

if __name__ == '__main__':
    app.run()
