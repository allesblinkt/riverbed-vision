class Camera(object):
    resx = 1080.0  # image width (in pixels). Enter transposed!
    resy = 1920.0  # image height (in pixels). Enter transposed!
                   
    viewx = 53.0 * 2.0  # view width (in cnc units = mm). Enter transposed!
    viewy = 95.0 * 2.0  # view height (in cnc units = mm). Enter transposed!
                        
    view_transpose = True
    view_flip = True

    offset_x = 45.0  # -3.0
    offset_y = -6.0  # +62.00

    v4l_params_1 = {
        'brightness': 128,
        'contrast': 128,
        'saturation': 128,
        'white_balance_temperature_auto': 0,
        'gain': 0,
        'power_line_frequency': 1,   # 50 Hz
        'white_balance_temperature': 5000,  # TODO: check
        'sharpness': 128,
        'backlight_compensation': 0,
        'exposure_auto': 1,   # TODO: check
        'exposure_absolute': 100,
        'exposure_auto_priority': 0,
        'pan_absolute': 0,
        'tilt_absolute': 0,
        'focus_auto': 0,
    }

    v4l_params_2 = {
        'focus_absolute': 25,
        'zoom_absolute': 100,
    }

    grab_dummy_frames = 3

class Machine(object):

    # CONTROL_HOSTNAME = 'localhost'
    CONTROL_HOSTNAME = '10.0.42.42'

    head_arm_length = 0.0
    lift_up_tries = 2
    lift_up_jitter_rad = 3
    lift_down_extra_z_down = 2.5
    lift_down_eject_dwell = 15

class Brain(object):

    init_x = 100
    init_y = 100
    init_e = 90
    init_feedrate = 32000

class StoneMap(object):

    size = 3770, 1730
