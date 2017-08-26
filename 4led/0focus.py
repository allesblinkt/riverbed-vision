#!/usr/bin/env python3

import cv2
import numpy as np

stem = 'depth_r2/grab_0000_0000_l0'

im0e = cv2.imread('empty_l0.jpg')

im0 = cv2.imread('%s_f30.jpg' % stem)
im1 = cv2.imread('%s_f60.jpg' % stem)

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

cv2.imshow('chain', chain)

cv2.waitKey(0)
