#!/usr/bin/env python3
import sys

from flask import Flask, render_template, send_file
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

@app.route('/map.png')
def map_image():
    return send_file(open('/tmp/stonemap.png', 'rb'), mimetype='image/png', cache_timeout=1)

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

@app.route('/speed_normal', methods=['POST'])
def speed_normal():
    status.write(speed='normal')
    return 'ok'

@app.route('/speed_slow', methods=['POST'])
def speed_slow():
    status.write(speed='slow')
    return 'ok'

if __name__ == '__main__':
    app.run()
