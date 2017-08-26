#!/usr/bin/env python3

import cv2
import numpy as np
from scipy.stats import threshold
from scipy.signal import convolve2d
import glob

USE_EMPTY = True

DATADIR = 'grab_r2'


def sharpen(img):
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    return cv2.addWeighted(img, 3, blur, -2, 0)


def minimum(a, b, c, d):
    m1 = np.minimum(a, b)
    m2 = np.minimum(c, d)
    return np.minimum(m1, m2)


def maximum(a, b, c, d):
    m1 = np.maximum(a, b)
    m2 = np.maximum(c, d)
    return np.maximum(m1, m2)


def callback(x):
    pass

cv2.namedWindow('control')
cv2.createTrackbar('s_min', 'control',  60, 1000, callback)
cv2.createTrackbar('s_max', 'control', 558, 1000, callback)
cv2.createTrackbar('v_min', 'control', 326, 1000, callback)
cv2.createTrackbar('v_max', 'control', 950, 1000, callback)

if USE_EMPTY:
    im0e = cv2.imread('empty_l0.jpg')
    im1e = cv2.imread('empty_l1.jpg')
    im2e = cv2.imread('empty_l2.jpg')
    im3e = cv2.imread('empty_l3.jpg')
    im0e = cv2.resize(im0e, (1080//4, 1920//4))
    im1e = cv2.resize(im1e, (1080//4, 1920//4))
    im2e = cv2.resize(im2e, (1080//4, 1920//4))
    im3e = cv2.resize(im3e, (1080//4, 1920//4))



files = sorted(glob.glob('%s/*_l0.jpg' % DATADIR))
fileindex = 0

while True:

    filename = files[fileindex]
    stem = filename[:-7]
    print(stem)

    im0 = cv2.imread('%s_l0.jpg' % stem)
    im1 = cv2.imread('%s_l1.jpg' % stem)
    im2 = cv2.imread('%s_l2.jpg' % stem)
    im3 = cv2.imread('%s_l3.jpg' % stem)

    im0 = cv2.resize(im0, (1080//4, 1920//4))
    im1 = cv2.resize(im1, (1080//4, 1920//4))
    im2 = cv2.resize(im2, (1080//4, 1920//4))
    im3 = cv2.resize(im3, (1080//4, 1920//4))

    if USE_EMPTY:
        im0 = im0.astype(np.float32) / (im0e.astype(np.float32) + 0.01)
        im1 = im1.astype(np.float32) / (im1e.astype(np.float32) + 0.01)
        im2 = im2.astype(np.float32) / (im2e.astype(np.float32) + 0.01)
        im3 = im3.astype(np.float32) / (im3e.astype(np.float32) + 0.01)

    im0h = cv2.cvtColor(im0, cv2.COLOR_BGR2HSV)[:,:,0]
    im1h = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)[:,:,0]
    im2h = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)[:,:,0]
    im3h = cv2.cvtColor(im3, cv2.COLOR_BGR2HSV)[:,:,0]

    im0s = cv2.cvtColor(im0, cv2.COLOR_BGR2HSV)[:,:,1]
    im1s = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)[:,:,1]
    im2s = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)[:,:,1]
    im3s = cv2.cvtColor(im3, cv2.COLOR_BGR2HSV)[:,:,1]

    im0v = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)[:,:,2]
    im1v = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)[:,:,2]
    im2v = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)[:,:,2]
    im3v = cv2.cvtColor(im3, cv2.COLOR_BGR2HSV)[:,:,2]

    s_min = minimum(im0s, im1s, im2s, im3s)
    s_max = maximum(im0s, im1s, im2s, im3s)

    s = np.ones_like(s_min) * 0.5
    s[s_min < cv2.getTrackbarPos('s_min','control') / 1000] = 1
    s[s_max > cv2.getTrackbarPos('s_max','control') / 1000] = 0
    s = cv2.medianBlur((s * 255).astype(np.uint8), 3) / 255

    # cv2.imshow('s_min', s_min)
    # cv2.imshow('s_max', s_max)
    cv2.imshow('s', s)

    v_min = minimum(im0v, im1v, im2v, im3v)
    v_max = maximum(im0v, im1v, im2v, im3v)

    v = np.ones_like(v_min) * 0.5
    v[v_min < cv2.getTrackbarPos('v_min','control') / 1000] = 0
    v[v_max > cv2.getTrackbarPos('v_max','control') / 1000] = 1
    v = cv2.medianBlur((v * 255).astype(np.uint8), 3) / 255

    # cv2.imshow('v_min', v_min)
    # cv2.imshow('v_max', v_max)
    cv2.imshow('v', v)

    r = np.ones_like(s) * 0.5
    r[s == 0] = 0
    r[v == 0] = 0
    r[s == 1] = 1
    r[v == 1] = 1
    cv2.imshow('r', r)

    k = cv2.waitKey(1)

    if k == 32:
        fileindex += 1

    if k == 27:
        break
