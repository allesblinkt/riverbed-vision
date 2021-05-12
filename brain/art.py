import math
from random import choice, random
from coloranalysis import compare_colors
from structure import compare_histograms
from utils import map_value, constrain

from log import makelog
log = makelog(__name__)

MAX_STAGE_MODE = 2  # How many stages / modes can we do (+1)
WORKAREA_START_X = 2400
FLOWERS = 9


flower_seeds = None
flower_luma_bins = None

# min_l, max_l = None, None    # Cache luma calculations
min_l, max_l = 0, 135    # Cache luma calculations


def find_flower_pos(stonemap, stone, center):
    radius, angle = 0.0, 0
    stone_dummy = stone.copy()

    # workarea_sel = [s for s in stonemap.stones if in_workarea(s)]

    while True:
        new_center = center[0] + math.cos(math.radians(angle)) * radius, center[1] + math.sin(math.radians(angle)) * radius
        stone_dummy.center = new_center
        stone_dummy.angle = angle % 180

        if in_workarea(stone_dummy) and stonemap.can_put(stone_dummy, border=2):
            return new_center, angle % 180

        angle += (137.50776405 / 10.0)
        if angle > 360.0:
            angle -= 360.0
            radius += 1.0


def find_best_match(stone, selection, bucket_size_c=10, bucket_size_s=50):
    min_colors = sorted(selection, key=lambda x: compare_colors(x.color, stone.color))
    min_structures = sorted(selection, key=lambda x: compare_histograms(x.structure, stone.structure))

    # for s in selection:
    #     log.info(stone.color)

    #     log.info(s.color)
    #     log.info(compare_colors(stone.color, s.color))
    #     log.info("===")

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


def art_step(stonemap):
    if stonemap.stage is not None:
        stage_mode, stage_step, stage1_y, stage1_last, stage1_first = stonemap.stage
        # stage1_last = stonemap.stones[stage1_last_index] if stage1_last_index is not None else None   # TODO: First
    else:
        log.info('Created new stage')
        stage_mode = -1
        stage1_y = None
        stage1_last = None
        stage1_first = None
        stage_step = 0

    # Color range
    global min_l, max_l
    if min_l is None:
        min_l = min(stonemap.stones, key=lambda x: x.color[0]).color[0]
    if max_l is None:
        max_l = max(stonemap.stones, key=lambda x: x.color[0]).color[0]

    global flower_seeds
    global flower_luma_bins

    # Generate seeds
    if flower_seeds is None:
        max_x, max_y = stonemap.size[0], stonemap.size[1]

        flower_seeds = []

        for i in range(FLOWERS):
            margin_y = 180
            y = map_value(i, 0, FLOWERS - 1, margin_y, max_y - margin_y)

            if i % 2 == 0:
                x = max_x - 350.0
            else:
                x = max_x - 950.0

            flower_seeds.append((x, y))

    chosen_stone, new_center, new_angle = None, None, None

    # clean unusable holes
    stonemap.holes = [h for h in stonemap.holes if not in_workarea(h) and h.center[0] + h.size <= WORKAREA_START_X - (stonemap.maxstonesize + 10) * (stage_step + 1)]

    if stage_mode == -1:   # Unflag to clear stones
        sel = [s for s in stonemap.stones if not in_workarea(s) and s.center[0] + s.size[0] > WORKAREA_START_X - (stonemap.maxstonesize + 10) * (stage_step + 1) and s.center[0] + s.size[0] <= WORKAREA_START_X - (stonemap.maxstonesize + 10) * (stage_step)]

        if sel:
            for s in sel:
                s.flag = False
        stage_mode == 0
    if stage_mode == 0:   # Clear area
        sel = [s for s in stonemap.stones if not s.flag and not in_workarea(s) and s.center[0] + s.size[0] > WORKAREA_START_X - (stonemap.maxstonesize + 10) * (stage_step + 1) and s.center[0] + s.size[0] <= WORKAREA_START_X - (stonemap.maxstonesize + 10) * (stage_step)]
        if sel:
            s = sel[0]
            chosen_stone = s

            l_norm =  map_value(s.color[0], min_l, max_l, 0.0, 1.0)
            l_norm = pow(l_norm, 2.0)
            bucket = map_value(l_norm, 0.0, 1.0, 0.0, len(flower_seeds) + 1)
            bucket = constrain(int(bucket), 0, len(flower_seeds) - 1)
            new_center, new_angle = find_flower_pos(stonemap, s, flower_seeds[bucket])

    elif stage_mode == 1:  # Fill line
        untouched_sel = [s for s in stonemap.stones if s.center[0] + s.size[0] <= WORKAREA_START_X - (stonemap.maxstonesize + 10) * (stage_step + 1)]
        workarea_sel = [s for s in stonemap.stones if in_workarea(s)]

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
                if stage1_first is not None:
                    s = find_most_distant_color(stage1_first, sel)
                else:
                    s = choice(sel)

                stage1_first = s
                stage1_last = s
            else:
                s = find_best_match(stage1_first, sel)
                stage1_y += stage1_last.size[1] + s.size[1] + 5
                stage1_last = s
            chosen_stone = s
            new_angle = 0
            x = WORKAREA_START_X - (stonemap.maxstonesize + 10) * (stage_step + 0.5)

            stone_dummy = chosen_stone.copy()
            stone_dummy.center = x, stage1_y
            stone_dummy.angle = new_angle
            # ignore stones exactly below me (they are from previous activity)
            ignore = [s for s in stonemap.stones if s.center[0] == x and s.center[1] < stage1_y]
            while not stonemap.can_put(stone_dummy, tight=True, ignore=ignore):
                stage1_y += 5
                stone_dummy.center = x, stage1_y

            new_center = x, stage1_y
            if stage1_y > 1520:
                stage1_y = None
                stage_step += 1
                stage_mode = -1

    elif stage_mode == 2:   # Done
        pass

    if chosen_stone is not None:
        force_advance = False
        log.debug('Art stage mode %d: stone %s => new center: %s, new angle: %s', stage_mode, str(chosen_stone), str(new_center), str(new_angle))
    else:
        force_advance = True
        stage_mode = min(stage_mode + 1, MAX_STAGE_MODE)
        log.debug('Art stage mode %d: None', stage_mode)

    stage = stage_mode, stage_step, stage1_y, stage1_last, stage1_first   # Do not store in map

    return chosen_stone, new_center, new_angle, stage, force_advance
