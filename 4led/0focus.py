#!/usr/bin/env python3

import cv2
import numpy as np
import glob

DATADIR = 'grab_depth_r3'

im0e = cv2.imread('empty_l0.jpg')

files = sorted(glob.glob('%s/*_l0_f30.jpg' % DATADIR))

for filename in files:
    stem = filename[:-8]
    print(stem)

    im0 = cv2.imread('%s_f30.jpg' % stem)
    im1 = cv2.imread('%s_f60.jpg' % stem)

    cv2.imshow('im0', cv2.resize(im0, dsize=(0, 0), fx=0.5, fy=0.5))

    im0 = im0.astype(np.float32) / (im0e.astype(np.float32) + 0.01)
    im1 = im1.astype(np.float32) / (im0e.astype(np.float32) + 0.01)

    im_s = cv2.cvtColor(im0, cv2.COLOR_BGR2HSV)[:,:,1]
    im_v = cv2.cvtColor(im0, cv2.COLOR_BGR2HSV)[:,:,2]

    # cv2.imshow('im0', im0)
    # cv2.imshow('im1', im1)

    im_sobel1 = cv2.Sobel(im0, ddepth=-1, dx=0, dy=1, ksize=3)
    im_sobel2 = cv2.Sobel(im0, ddepth=-1, dx=1, dy=0, ksize=3)
    im_sobel = cv2.magnitude(im_sobel1 * 2, im_sobel2 * 2)
    im_sobel = cv2.cvtColor(im_sobel, cv2.COLOR_BGR2HSV)[:,:,2]
    im_sobel[im_sobel > 0.3] = 1
    im_sobel[im_sobel <= 0.3] = 0

    im_diff = np.absolute(im0 - im1)
    im_diff = cv2.cvtColor(im_diff, cv2.COLOR_BGR2HSV)[:,:,2]
    im_diff[im_diff > 0.07] = 1

    chain = np.maximum(im_sobel, im_diff)

    chain[im_v < 0.51] = 1
    chain[im_s > 0.14] = 1

    chain = cv2.medianBlur((chain * 255).astype(np.uint8), 5) / 255

    cv2.imshow('chain', cv2.resize(chain, dsize=(0, 0), fx=0.5, fy=0.5))

    imc = im0.copy()
    _, contours, _ = cv2.findContours(chain.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
    print('contours', cnt)
    cv2.imshow('imc', cv2.resize(imc, dsize=(0, 0), fx=0.5, fy=0.5))

    k = cv2.waitKey(0)

    if k == 27:
        break
