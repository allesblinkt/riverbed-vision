from log import log
import math
import numpy as np
from random import uniform, choice
from coloranalysis import compare_colors
from structure import compare_histograms
from utils import map_value, constrain

STAGE = 0

MAX_STAGE = 2

THRESH = 2500

flower_seeds = None


def find_flower_pos(map, stone, center):
    radius, angle = 0.0, 0
    stone2 = stone.copy()
    while True:
        x, y = center[0] + math.cos(math.radians(angle)) * radius, center[1] + math.sin(math.radians(angle)) * radius
        stone2.center = x, y
        stone2.angle = angle % 180
        if map.can_put(stone2): # TODO: optimize by checking only against workarea stones?
            return (x, y), angle % 180
        angle += 137.50776405
        if angle > 360:
            angle -= 360
            radius += 2.0


def find_best_match(stone, selection, bucket_size_c=10, bucket_size_s=200):
    min_colors = sorted(selection, key=lambda x: compare_colors(x.color, stone.color))
    min_structures = sorted(selection, key=lambda x: compare_histograms(x.structure, stone.structure))

    color_set = set(min_colors[:bucket_size_c])
    structure_set = set(min_structures[:bucket_size_s])

    intersection_list = list(color_set.intersection(structure_set))

    if len(intersection_list) > 0:
        print "!!!!!!!!!!!!1 Found in set intersection"
        return choice(intersection_list)

    return min_colors[0]


def find_most_distant_color(stone, selection):
    s = max(selection, key=lambda x: compare_colors(x.color, stone.color))
    return s


def in_workarea(stone):
    return stone.center[0] > THRESH

stage1_y = None
stage1_last = None
stage_step = 0
min_l, max_l = None, None

def art_step(map):
    global STAGE
    global stage1_y, stage1_last
    global stage_step

    if map.stage is not None:
        STAGE, stage_step, stage1_y, stage1_last_index = map.stage
        stage1_last = map.stones[stage1_last_index] if stage1_last_index is not None else None

    # Color range
    global min_l, max_l
    if min_l is None:
        min_l = min(map.stones, key=lambda x: x.color[0]).color[0]
    if max_l is None:
        max_l = max(map.stones, key=lambda x: x.color[0]).color[0]

    global flower_seeds

    if flower_seeds is None:
        max_x, max_y = map.size[0], map.size[1]

        flower_seeds = []

        for i in range(9):
            y = map_value(i, 0, 8, 300, max_y - 300)

            if i % 2 == 0:
                x = max_x - 340.0
            else:
                x = max_x - 880.0

            flower_seeds.append((x, y))

    index, new_center, new_angle = None, None, None

    # clean unusable holes
    map.holes = [h for h in map.holes if not in_workarea(h) and h.center[0] + h.size <= THRESH - (map.maxstonesize + 10) * (stage_step + 1)]

    if STAGE == 0:
        sel = [s for s in map.stones if not s.flag and not in_workarea(s) and s.center[0] + s.size[0] > THRESH - (map.maxstonesize + 10) * (stage_step + 1) and s.center[0] + s.size[0] <= THRESH - (map.maxstonesize + 10) * (stage_step) ]
        if sel:
            s = sel[0]
            index = s.index

            bucket = map_value(s.color[0], min_l, max_l, 0, len(flower_seeds) + 1)
            bucket = constrain(int(bucket), 0, len(flower_seeds) - 1)
            new_center, new_angle = find_flower_pos(map, s, flower_seeds[bucket])

    elif STAGE == 1:
        sel = [s for s in map.stones if not s.flag and (in_workarea(s) or s.center[0] + s.size[0] <= THRESH - (map.maxstonesize + 10) * (stage_step + 1))]
        if sel:
            if stage1_y is None:
                # first run of this stage
                stage1_y = 50
                # pick random stone
                if stage1_last is not None:
                    s = find_most_distant_color(stage1_last, sel)
                else:
                    s = choice(sel)

                stage1_last = s
            else:
                s = find_best_match(stage1_last, sel)
                stage1_y += stage1_last.size[1] + s.size[1] + 5
                stage1_last = s
            index = s.index
            new_angle = 0
            x = THRESH - (map.maxstonesize + 10) * (stage_step + 0.5)
            new_center = x, stage1_y
            if stage1_y > 1650:
                stage1_y = None
                stage_step += 1
                STAGE = 0

    elif STAGE == 2:
        pass

    if index is not None:
        log.debug('Art stage %d: stone %s => new center: %s, new angle: %s', STAGE, index, str(new_center), str(new_angle))
    else:
        STAGE = min(STAGE + 1, MAX_STAGE)
        log.debug('Art stage %d: None', STAGE)

    map.stage = STAGE, stage_step, stage1_y, stage1_last.index is not None if stage1_last else None

    return index, new_center, new_angle
