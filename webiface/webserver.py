#!/usr/bin/env python3
import sys

from flask import Flask, render_template
from flask.json import jsonify

import launcher
import status

app = Flask(__name__)

@app.route('/')
def root():
    return render_template('page.html')

@app.route('/status.json')
def status_json():
    s = status.read()
    s['alive'] = launcher.is_alive()
    return jsonify(s)

@app.route('/start', methods=['POST'])
def start():
    launcher.run()
    return 'ok'

@app.route('/kill', methods=['POST'])
def kill():
    launcher.kill()
    return 'ok'

@app.route('/pause', methods=['POST'])
def pause():
    status.write(state='paused')
    return 'ok'

@app.route('/unpause', methods=['POST'])
def unpause():
    status.write(state='working')
    return 'ok'

if __name__ == '__main__':
    app.run()
