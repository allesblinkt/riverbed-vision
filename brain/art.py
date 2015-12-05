from log import log
import math
import numpy as np
from random import uniform

STAGE = 0

MAX_STAGE = 2

flower_seeds = [
    (333.0,  333.0), (666.0 , 333.0),
    (333.0,  666.0), (666.0,  666.0),
    (333.0,  999.0), (666.0,  999.0),
    (333.0, 1333.0), (666.0, 1333.0),
    (333.0, 1666.0), (666.0, 1666.0),
]

def find_flower_pos(map, stone, center):
    radius, angle = 0.0, 0
    stone2 = stone.copy()
    while True:
        x, y = center[0] + math.cos(math.radians(angle)) * radius, center[1] + math.sin(math.radians(angle)) * radius
        stone2.center = x, y
        stone2.angle = angle % 180
        if map.can_put(stone2):
            return (x, y), angle % 180
        angle += 137.508
        if angle > 360:
            angle -= 360
            radius += 1.0

def art_step(map):
    global STAGE
    index, new_center, new_angle = None, None, None

    if STAGE == 0:
        for i, s in enumerate(map.stones):
            if s.center[1] > 1800 and s.angle != 90:
                index = i
                new_angle = 90
                bucket = int(s.color[0] * 10 / 255)
                new_center, new_angle = find_flower_pos(map, s, flower_seeds[bucket])
                break

    if STAGE == 1:
        sel = [s for s in map.stones if s.center[1] <= 1800]
        if sel:
            s = min(sel, key=lambda x: x.color[0])
            index = s.index
            new_angle = 90
            new_center = 1000 + s.color[0] * 10, 1900

    if STAGE == 2:
        pass

    if index is not None:
        log.debug('Art stage %d: stone %s => new center: %s, new angle: %s', STAGE, index, str(new_center), str(new_angle))
    else:
        STAGE = min(STAGE + 1, MAX_STAGE)
        log.debug('Art stage %d: None', STAGE)

    return index, new_center, new_angle
