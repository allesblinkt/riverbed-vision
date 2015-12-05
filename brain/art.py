from log import log

STATE = 0

def art_step(map):
    stones = map.stones
    new_center, new_angle = None, None
    for i, s in enumerate(stones):

        if STATE == 0:

            if s.center[1] > 1800 and s.angle != 90:
                new_angle = 90
                break

    log.debug('Art step: stone %d - new center: %s, new angle: %s', i, str(new_center), str(new_angle))
    return i, new_center, new_angle
