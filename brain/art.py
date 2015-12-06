from log import log
import math
import numpy as np
from random import uniform, choice
from coloranalysis import compare_colors
from structure import compare_histograms

STAGE = 0

MAX_STAGE = 2

flower_seeds = [
    (3000 + 666.0,  333.0), (3000 + 333.0 , 333.0),
    (3000 + 666.0,  666.0), (3000 + 333.0,  666.0),
    (3000 + 666.0,  999.0), (3000 + 333.0,  999.0),
    (3000 + 666.0, 1333.0), (3000 + 333.0, 1333.0),
    (3000 + 666.0, 1666.0), (3000 + 333.0, 1666.0),
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
        angle += 137.50776405
        if angle > 360:
            angle -= 360
            radius += 10.0

def in_workarea(stone):
    return stone.center[0] >= 3000

stage1_x = None
stage1_last = None
stage_step = 1

def art_step(map):
    global STAGE
    global stage1_x, stage1_last
    global stage_step

    index, new_center, new_angle = None, None, None

    if STAGE == 0:
        sel = [s for s in map.stones if not in_workarea(s) and s.center[1] + s.size[0] > 2000 - (map.maxstonesize + 10) * stage_step and not s.done]
        if sel:
            s = sel[0]
            index = s.index
            bucket = int(s.color[0] * 10 / 255)
            new_center, new_angle = find_flower_pos(map, s, flower_seeds[bucket])

    if STAGE == 1:
        sel = [s for s in map.stones if in_workarea(s) or s.center[1] + s.size[0] <= 2000 - (map.maxstonesize + 10) * stage_step]
        if sel:
            if stage1_x is None:
                # first run of this stage
                stage1_x = 3000
                # pick random stone
                s = choice(sel)
                stage1_last = s
            else:
                s = min(sel, key=lambda x: compare_colors(x.color, stage1_last.color) * compare_histograms(x.structure, stage1_last.structure) )
                stage1_last = s
            s.done = True
            index = s.index
            new_angle = 90
            y = 2000 - (map.maxstonesize + 10) * (stage_step - 0.5)
            new_center = stage1_x, y
            stage1_x -= s.size[1] * 2.0 + 5 # TODO: fixme - not entirely correct
            if stage1_x < 100:
                stage1_x = None
                stage_step += 1
                STAGE = 0

    if STAGE == 2:
        pass

    if index is not None:
        log.debug('Art stage %d: stone %s => new center: %s, new angle: %s', STAGE, index, str(new_center), str(new_angle))
    else:
        STAGE = min(STAGE + 1, MAX_STAGE)
        log.debug('Art stage %d: None', STAGE)

    return index, new_center, new_angle
