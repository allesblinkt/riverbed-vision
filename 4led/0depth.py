#!/usr/bin/env python3

import cv2
import numpy as np
from scipy.stats import threshold
from scipy.signal import convolve2d

DATASET = 'grab_0000_0000'

USE_EMPTY = True

im1 = cv2.imread(DATASET + '_l1.jpg')
im2 = cv2.imread(DATASET + '_l2.jpg')
im3 = cv2.imread(DATASET + '_l3.jpg')

if USE_EMPTY:
    im1e = cv2.imread('empty_' + DATASET + '_l1.jpg')
    im2e = cv2.imread('empty_' + DATASET + '_l2.jpg')
    im3e = cv2.imread('empty_' + DATASET + '_l3.jpg')
    im1 = 255 - cv2.subtract(im1e, im1)
    im2 = 255 - cv2.subtract(im2e, im2)
    im3 = 255 - cv2.subtract(im3e, im3)

im1s = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)[:,:,1].astype(float) / 255
im2s = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)[:,:,1].astype(float) / 255
im3s = cv2.cvtColor(im3, cv2.COLOR_BGR2HSV)[:,:,1].astype(float) / 255

im1v = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)[:,:,2].astype(float) / 255
im2v = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)[:,:,2].astype(float) / 255
im3v = cv2.cvtColor(im3, cv2.COLOR_BGR2HSV)[:,:,2].astype(float) / 255

im1g = im1v * (1 - im1s)
im2g = im2v * (1 - im2s)
im3g = im3v * (1 - im3s)

cv2.imshow('im1g', im1g)
cv2.imshow('im2g', im2g)
cv2.imshow('im3g', im3g)

im_min = np.minimum(np.minimum(im1g, im2g), im3g)
im_avg = (im1g + im2g + im3g) / 3.0
im_max = np.maximum(np.maximum(im1g, im2g), im3g)

im_max2 = im_max * im_max

cv2.imshow('im_min', im_min)
cv2.imshow('im_avg', im_avg)
cv2.imshow('im_max', im_max)
cv2.imshow('im_max2', im_max2)

ones = np.ones_like(im_max2)
max_thresh = threshold(im_max2, None, ones * 0.8, 1)

cv2.imshow('max_thresh', max_thresh)

shadow = max_thresh - im_min

cv2.imshow('shadow', shadow)

"""
im1r = im1g / im_max
im2r = im2g / im_max
im3r = im3g / im_max

cv2.imshow('im1r', im1r)
cv2.imshow('im2r', im2r)
cv2.imshow('im3r', im3r)

imr = im1r * im2r * im3r

cv2.imshow('imr', imr)

imd = im_max - imr

cv2.imshow('imd', imd)
"""

# thresh = 0.65
# im1t = threshold(threshold(im1r, threshmin=thresh, newval=0), threshmax=thresh, newval=1)
# im2t = threshold(threshold(im2r, threshmin=thresh, newval=0), threshmax=thresh, newval=1)
# im3t = threshold(threshold(im3r, threshmin=thresh, newval=0), threshmax=thresh, newval=1)

# imt = im1t * im2t * im3t

# cv2.imshow('im1t', im1t)
# cv2.imshow('im2t', im2t)
# cv2.imshow('im3t', im3t)
# cv2.imshow('imt', imt)

"""
im1c = convolve2d(im1r, [[-1, -1, -1, -1], [-1, 0, 0, -1], [+1, 0, 0, +1], [+1, +1, +1, +1]], mode='same') # down
im2c = convolve2d(im2r, [[+1, +1, -1, -1], [+1, 0, 0, -1], [+1, 0, 0, -1], [+1, +1, -1, -1]], mode='same') # left
im3c = convolve2d(im3r, [[+1, +1, +1, +1], [-1, 0, 0, +1], [-1, 0, 0, +1], [-1, -1, -1, -1]], mode='same') # top-right

# TODO: set border 2px to 0

imc = im1c + im2c + im3c

cv2.imshow('im1c', im1c)
cv2.imshow('im2c', im2c)
cv2.imshow('im3c', im3c)

cv2.imshow('imc', imc)
"""

cv2.waitKey(0)