from log import log

STATE = 0

def art_step(map):
    stones = map.stones
    index, new_center, new_angle = None, None, None

    for i, s in enumerate(stones):

        if STATE == 0:

            if s.center[1] > 1800 and s.angle != 90:
                index = i
                new_angle = 90
                new_center = s.color[1] * 2, s.color[0] * 4
                break


    if index is not None:
        log.debug('Art step: stone %s => new center: %s, new angle: %s', index, str(new_center), str(new_angle))
    else:
        log.debug('Art step: None')

    return index, new_center, new_angle
