class Camera(object):

    resx = 720.0  # image width (in pixels). Transposed!
    resy = 1280.0  # image height (in pixels). Transposed!
    viewx = 39.0 * 2.0  # view width (in cnc units = mm). Transposed!
    viewy = 69.0 * 2.0  # view height (in cnc units = mm). Transposed!
    flipall = True
    offset_x = -3.0
    offset_y = 62.00  # used to be -3, +66

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
        'exposure_absolute': 250,    # TODO: check
        'exposure_auto_priority': 0,
        'pan_absolute': 0,
        'tilt_absolute': 0,
        'focus_auto': 0,
    }

    v4l_params_2 = {
        'focus_absolute': 25,
        'zoom_absolute': 100,
    }

class Machine(object):

    head_length = 0.0