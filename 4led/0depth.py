#!/usr/bin/env python3

import cv2
import numpy as np
from scipy.stats import threshold
from scipy.signal import convolve2d

def sharpen(img):
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    return cv2.addWeighted(img, 3, blur, -2, 0)


DATADIR = 'grab_r2'
DATASET = 'grab_0000_0000'

USE_EMPTY = True

im1 = cv2.imread('%s/%s_l1.jpg' % (DATADIR, DATASET))
im2 = cv2.imread('%s/%s_l2.jpg' % (DATADIR, DATASET))
im3 = cv2.imread('%s/%s_l3.jpg' % (DATADIR, DATASET))

if USE_EMPTY:
    im1e = cv2.imread('empty_l1.jpg')
    im2e = cv2.imread('empty_l2.jpg')
    im3e = cv2.imread('empty_l3.jpg')
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

# cv2.imshow('im1g', im1g)
# cv2.imshow('im2g', im2g)
# cv2.imshow('im3g', im3g)

im_min = np.minimum(np.minimum(im1g, im2g), im3g)
im_avg = (im1g + im2g + im3g) / 3.0
im_max = np.maximum(np.maximum(im1g, im2g), im3g)

# im_min = sharpen(im_min)
# im_avg = sharpen(im_avg)
# im_max = sharpen(im_max)

# cv2.imshow('im_min', im_min)
# cv2.imshow('im_avg', im_avg)
# cv2.imshow('im_max', im_max)

thresh1 = cv2.adaptiveThreshold((im1g * 255).astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 9) / 255
thresh2 = cv2.adaptiveThreshold((im2g * 255).astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 9) / 255
thresh3 = cv2.adaptiveThreshold((im3g * 255).astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 9) / 255

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
thresh1 = cv2.erode(thresh1, kernel, iterations=3)
thresh2 = cv2.erode(thresh2, kernel, iterations=3)
thresh3 = cv2.erode(thresh3, kernel, iterations=3)

# cv2.imshow('thresh1', thresh1)
# cv2.imshow('thresh2', thresh2)
# cv2.imshow('thresh3', thresh3)

# thresh_min = cv2.adaptiveThreshold((im_min * 255).astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 9) / 255
# thresh_avg = cv2.adaptiveThreshold((im_avg * 255).astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 9) / 255
# thresh_max = cv2.adaptiveThreshold((im_max * 255).astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 9) / 255

# cv2.imshow('thresh_min', thresh_min)
# cv2.imshow('thresh_avg', thresh_avg)
# cv2.imshow('thresh_max', thresh_max)

sat = im1s + im2s + im3s
thresh = thresh1 * thresh2 * thresh3
thresh[im_min > 0.75] = 1
thresh[im_max < 0.65] = 0
thresh[sat > 0.4] = 0
# cv2.imshow('thresh', thresh)

final = thresh
final = cv2.dilate(final, kernel, iterations=2)
final = cv2.medianBlur((final * 255).astype(np.uint8), 3) / 255
cv2.imshow('final', final)

imc = im1.copy()
_, contours, _ = cv2.findContours((255 - final * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = 0
for i in range(len(contours)):
    ec, es, ea = cv2.minAreaRect(contours[i])
    if es[0] < 40 or es[1] < 40:
        continue
    cv2.drawContours(imc, contours, i, (255, 0, 0), -1)
    fit_center = (int(ec[0]), int(ec[1]))
    fit_dim = (int(es[0]) // 2, int(es[1]) // 2)
    fit_angle = int(ea)
    cv2.ellipse(imc, fit_center, fit_dim, fit_angle, 0, 360, (0, 0, 255), 2)
    cnt += 1
cv2.imshow('imc (%d)' % cnt, imc)

cv2.waitKey(0)
