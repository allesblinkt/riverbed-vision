import math
from random import choice, random
from coloranalysis import compare_colors
from structure import compare_histograms
from utils import map_value, constrain

from log import makelog
log = makelog(__name__)

MAX_STAGE_MODE = 2  # How many stages / modes can we do (+1)
WORKAREA_START_X = 2100

flower_seeds = None
min_l, max_l = None, None    # Cache luma calculations

flower_cache = {}

def find_flower_pos(map, stone, center):
    global flower_cache
    if center in flower_cache:
        radius, angle = flower_cache[center]
    else:
        radius, angle = 0.0, 0
    stone2 = stone.copy()

    workarea_sel = [s for s in map.stones if in_workarea(s)]

    while True:
        x, y = center[0] + math.cos(math.radians(angle)) * radius, center[1] + math.sin(math.radians(angle)) * radius
        stone2.center = x, y
        stone2.angle = angle % 180
        if in_workarea(stone2) and map.can_put_list(stone2, workarea_sel):
            return (x, y), angle % 180
        angle += (137.50776405 / 5.0)
        if angle > 360:
            angle -= 360
            radius += 2.0
    flower_cache[center] = radius, angle


def find_best_match(stone, selection, bucket_size_c=10, bucket_size_s=200):
    min_colors = sorted(selection, key=lambda x: compare_colors(x.color, stone.color))
    min_structures = sorted(selection, key=lambda x: compare_histograms(x.structure, stone.structure))

    color_set = set(min_colors[:bucket_size_c])
    structure_set = set(min_structures[:bucket_size_s])

    intersection_list = list(color_set.intersection(structure_set))

    if len(intersection_list) > 0:
        log.debug('Found in intersection set')
        return choice(intersection_list)

    return min_colors[0]


def find_most_distant_color(stone, selection):
    s = max(selection, key=lambda x: compare_colors(x.color, stone.color))
    return s


def in_workarea(stone):
    return stone.center[0] > WORKAREA_START_X


def atan_angle(p1, p2):
    return math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))

def art_step(map):
    if map.stage is not None:
        stage_mode, stage_step, stage1_y, stage1_last_index = map.stage
        stage1_last = map.stones[stage1_last_index] if stage1_last_index is not None else None
    else:
        stage_mode = 0
        stage1_y = None
        stage1_last = None
        stage_step = 0

    # Color range
    global min_l, max_l
    if min_l is None:
        min_l = min(map.stones, key=lambda x: x.color[0]).color[0]
    if max_l is None:
        max_l = max(map.stones, key=lambda x: x.color[0]).color[0]

    global flower_seeds

    # Generate seeds
    if flower_seeds is None:
        max_x, max_y = map.size[0], map.size[1]

        flower_seeds = []

        for i in range(9):
            margin_y = 180
            y = map_value(i, 0, 8, margin_y, max_y - margin_y)

            if i % 2 == 0:
                x = max_x - 400.0
            else:
                x = max_x - 1150.0

            flower_seeds.append((x, y))

    index, new_center, new_angle = None, None, None

    # clean unusable holes
    map.holes = [h for h in map.holes if not in_workarea(h) and h.center[0] + h.size <= WORKAREA_START_X - (map.maxstonesize + 10) * (stage_step + 1)]

    art_center = (WORKAREA_START_X / 2, map.size[1] / 2)

    if stage_mode == 0:   # Clear area
        sel = [s for s in map.stones if not s.flag and not in_workarea(s) and s.center[0] + s.size[0] > WORKAREA_START_X - (map.maxstonesize + 10) * (stage_step + 1) and s.center[0] + s.size[0] <= WORKAREA_START_X - (map.maxstonesize + 10) * (stage_step) ]
        if sel:
            s = sel[0]
            index = s.index

            bucket = map_value(s.color[0], min_l, max_l, 0, len(flower_seeds) + 1)
            bucket = constrain(int(bucket), 0, len(flower_seeds) - 1)
            new_center, new_angle = find_flower_pos(map, s, flower_seeds[bucket])

    elif stage_mode == 1:  # Fill line
        untouched_sel = [s for s in map.stones if s.center[0] + s.size[0] <= WORKAREA_START_X - (map.maxstonesize + 10) * (stage_step + 1)]
        workarea_sel = [s for s in map.stones if in_workarea(s)]

        max_fill = 2000
        rand_thresh = max(max_fill - len(workarea_sel), 0) / float(max_fill)

        if len(workarea_sel) > max_fill * 0.5 and random() > rand_thresh:
            total_sel = workarea_sel
        else:
            total_sel = workarea_sel + untouched_sel

        total_sel = workarea_sel + untouched_sel

        sel = [s for s in total_sel if not s.flag]

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
                rc = abs(math.cos(math.radians(stage1_last.angle)))
                rs = abs(math.sin(math.radians(stage1_last.angle)))
                a, b = stage1_last.size[1] * rc, stage1_last.size[0] * rs
                c, d = s.size[1] * rc, s.size[0] * rs
                e, f = math.sqrt(a * a + b * b), math.sqrt(c * c + d * d)
                stage1_y += e + f + 5
                stage1_last = s
            index = s.index
            x = WORKAREA_START_X - (map.maxstonesize + 10) * (stage_step + 0.5)
            new_center = x, stage1_y
            new_angle = atan_angle(art_center, new_center)
            if stage1_y > 1650:
                stage1_y = None
                stage_step += 1
                stage_mode = 0

    elif stage_mode == 2:   # Done
        pass

    if index is not None:
        force_advance = False
        log.debug('Art stage mode %d: stone %s => new center: %s, new angle: %s', stage_mode, index, str(new_center), str(new_angle))
    else:
        force_advance = True
        stage_mode = min(stage_mode + 1, MAX_STAGE_MODE)
        log.debug('Art stage mode %d: None', stage_mode)

    stage = stage_mode, stage_step, stage1_y, stage1_last.index if stage1_last else None   # Do not store in map

    return index, new_center, new_angle, stage, force_advance
