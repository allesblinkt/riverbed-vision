#!/usr/bin/env python3

import cv2
import numpy as np
from scipy.stats import threshold

DATASET = 'b'

img = []

for i in range(4):
    im = cv2.imread('%s%d.jpg' % (DATASET, i))
    img.append(im)

avg01 = cv2.addWeighted(img[0], 0.5, img[1], 0.5, 0)
avg23 = cv2.addWeighted(img[2], 0.5, img[3], 0.5, 0)
avg = cv2.addWeighted(avg01, 0.5, avg23, 0.5, 0)

max01 = np.maximum(img[0], img[1])
max23 = np.maximum(img[2], img[3])
maxi = np.maximum(max01, max23)

ones = np.ones_like(maxi)

max_thresh = threshold(maxi, None, ones * 224, 255)

shadow = max_thresh - avg

for i in range(4):
    cv2.imshow('img[%d]' % i, img[i])

cv2.imshow('avg', avg)
cv2.imshow('maxi', maxi)
cv2.imshow('max_thresh', max_thresh)
cv2.imshow('shadow', shadow)

cv2.waitKey(0)
