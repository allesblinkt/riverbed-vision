import cv2
import numpy as np
import logging as log

import time
# afsdfa32
# 21  33


def bla(orig_color_img, blank_img):
    # log.debug('Start processing image: %s', frame_desc)
    # start_time = time.time()



    start = time.time()

    # subtract blank vignette
    color_img = 255 - cv2.subtract(blank_img, orig_color_img)
    half_img = cv2.resize(color_img, (color_img.shape[1]//2, color_img.shape[0]//2))
    image = half_img

    # Grayscale conversion, blurring, threshold
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_img, s_img, v_img = cv2.split(hsv_img)

    gray_s_img = cv2.GaussianBlur(255 - s_img, (15, 15), 0)
    gray_v_img = cv2.GaussianBlur(v_img, (5, 5), 0)

    thresh_v_img = cv2.adaptiveThreshold(gray_v_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 151, -35)
    thresh_s_img = cv2.adaptiveThreshold(gray_s_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 151, -15)

    thresh_v_sure_img = cv2.adaptiveThreshold(gray_v_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 151, -5)
    thresh_v_sure_img[gray_v_img > 252] = 0    # prevent adaptive runaway

    # Secondary static threshhold on saturation
    thresh_s_img[gray_s_img > 235] = 0

    # "AND" with relaxed thresshold of values
    thresh_s_img[thresh_v_img < 128] = 0

    # "OR" with tight thresshold of values
    thresh_s_img[thresh_v_sure_img > 128] = 255

    thresh_img = thresh_s_img

    # thresh_img = cv2.resize(thresh_img, (thresh_img.shape[1]*4, thresh_img.shape[0]*4), interpolation=cv2.INTER_CUBIC)
    #thresh_img = cv2.GaussianBlur(thresh_img, (19, 19), 0)
    _, thresh_img = cv2.threshold(thresh_img, 128, 255, cv2.THRESH_BINARY)

    # print((time.time() - start)  * 1000)
    # elapsed_b = (time.time() - start)  * 1000
    # print(elapsed_b)
    # print((elapsed_a/elapsed_b)*100.0)


    #cv2.imshow('orig_color_img', orig_color_img)
    cv2.imshow('image', color_img)
    cv2.imshow('h', h_img)
    cv2.imshow('s', s_img)
    cv2.imshow('v', v_img)
    cv2.imshow('gray_s', gray_s_img)
    cv2.imshow('gray_v', gray_v_img)
    cv2.imshow('thresh_s_img', thresh_s_img)
    cv2.imshow('thresh_v_img', thresh_v_img)
    cv2.imshow('thresh_v_sure_img', thresh_v_sure_img)
    cv2.imshow('thresh_img', thresh_img)

    # cv2.imshow('dingedi',1.0-dingedi_img)

    cv2.waitKey(200)
    # print("Yooo!")







