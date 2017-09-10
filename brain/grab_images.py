from brain import DummyMachine, Camera
import cv2
import sys

if __name__ == '__main__':
    machine = DummyMachine('10.0.42.42')
    camera = Camera(machine, index=1)
    machine.cam = camera
    # machine.control.home()
    # machine.control.go(x=100)

    while True:
        # machine.control.go(y=100)
        imgt = camera.grab(save=True, light_channel=0)
        img1 = camera.grab(save=True, light_channel=1)
        img2 = camera.grab(save=True, light_channel=2)
        img3 = camera.grab(save=True, light_channel=3)
        # img_all = camera.grab(save=True, light_channel=None)

        p_scalef = 2

        if True:
            img1 = cv2.resize(img1, (img1.shape[1] // p_scalef, img1.shape[0] // p_scalef))
            img2 = cv2.resize(img2, (img2.shape[1] // p_scalef, img2.shape[0] // p_scalef))
            img3 = cv2.resize(img3, (img3.shape[1] // p_scalef, img3.shape[0] // p_scalef))
            imgt = cv2.resize(imgt, (imgt.shape[1] // p_scalef, imgt.shape[0] // p_scalef))
            # img_all = cv2.resize(img_all, (img_all.shape[1] // p_scalef, img_all.shape[0] // p_scalef))

        cv2.imshow('1', img1)
        cv2.imshow('2', img2)
        cv2.imshow('3', img3)
        cv2.imshow('t', imgt)
        # cv2.imshow('all', img_all)

        key = cv2.waitKey(1)

        if key == ord('q'):
            sys.exit(1)
