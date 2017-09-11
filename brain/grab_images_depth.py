from brain import DummyMachine, Camera
import time

if __name__ == '__main__':
    machine = DummyMachine('10.0.42.42')
    camera = Camera(machine, index=0)
    machine.cam = camera

    # machine.control.go(y=100)
    focus_stack = camera.focus_stack

    for focus in focus_stack:
        camera.set_cam_parameter('focus_absolute', focus)
        suffix = '_f%d_manual' % (focus, )

        time.sleep(1)
        imgt = camera.grab(save=True, light_channel=0, suffix=suffix)
