import cv2
import numpy as np
import logging as log
import importlib
import time

import threshholding

blank = cv2.imread('blank.png')
orig_color_img = cv2.imread('../brain/map_offline/grab_0975_0828.jpg')
# orig_color_img = cv2.imread('../brain/map_offline/grab_0000_0000.jpg')
    # grab_0429_1380.jpg
    # grab_0429_1380.jpg
    # grab_1755_1173.jpg
    # grab_1911_0276.jpg,
#../brain/map_offline/grab_1131_0621.jpg
    # FIXED
    # grab_1365_1311.jpg

print(blank.shape)
print(orig_color_img.shape)


while True:
    try:
        importlib.reload(threshholding)
        threshholding.bla(orig_color_img, blank)
    except Exception as e:
        print("Booooooo!")
        print(e)
        time.sleep(0.2)
    
cv2.destroyAllWindows()