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

   
    # image = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))

    # Grayscale conversion, blurring, threshold
   # Grayscale conversion, blurring, threshold
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # h_img, s_img, v_img = cv2.split(hsv_img)   # TODO: use numpy?
    # h_img = hsv_img[:, :, 0]  # TODO: check
    s_img = hsv_img[:, :, 1]
    v_img = hsv_img[:, :, 2]
    h_img = hsv_img[:, :, 0]

    gray_s_img = cv2.GaussianBlur(255 - s_img, (15, 15), 0)
    gray_v_img = cv2.GaussianBlur(v_img, (5, 5), 0)

    thresh_v_img = cv2.adaptiveThreshold(gray_v_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 151, -15)
    thresh_s_img = cv2.adaptiveThreshold(gray_s_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 151, -0)
    thresh_v_img[gray_v_img > 240] = 0    # prevent adaptive runaway

    thresh_v_sure_img = cv2.adaptiveThreshold(gray_v_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 151, 25)
    thresh_v_sure_img[gray_v_img < 210] = 255    # prevent adaptive runaway
    thresh_v_sure_img[gray_v_img >= 210] = 0    # prevent adaptive runaway

    # Secondary static threshhold on saturation
    thresh_s_img[gray_s_img < 100] = 0

    # "AND" with relaxed thresshold of values
    thresh_s_img[thresh_v_img < 128] = 0

    # "OR" with tight thresshold of values
    thresh_s_img[thresh_v_sure_img > 128] = 255

    thresh_img = thresh_s_img


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







