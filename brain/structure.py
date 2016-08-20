#!/usr/bin/python

import cv2
import numpy as np

from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq


def lbp_histogram(img, radius=5):
    """ Computes the LBP Histogram for a given images luminance channel and returns it """

    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)    # Convert to lab for better perceptual accuracy
    gray_img = lab_img[:, :, 0]

    n_points = 8 * radius  # points to consider

    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')

    x = itemfreq(lbp.ravel())
    hist = x[:, 1] / sum(x[:, 1])

    return hist


def compare_histograms(a, b):
    arr_a = np.array(a, dtype=np.float32)
    arr_b = np.array(b, dtype=np.float32)
    score = cv2.compareHist(arr_a, arr_b, cv2.HISTCMP_CHISQR)

    return score


if __name__ == '__main__':
    import os
    import fnmatch
    import time
    import sys

    p = "stones/"   # looks here for pngs...

    pngfiles = []

    for file in os.listdir(p):
        if fnmatch.fnmatch(file, '*.png'):
            pngfiles.append(file)

    first_img = cv2.imread(os.path.join(p, pngfiles[0]), -1)
    first_hist = lbp_histogram(first_img)

    for fn in pngfiles:
        log.info('Loading %s', fn)
        image = cv2.imread(os.path.join(p, fn), -1)
        (h, w) = image.shape[:2]

        t = time.time()
        hist = lbp_histogram(image)
        log.info('Time for lbp_histogram: %.3f', (time.time() - t))

        t = time.time()
        for i in range(1000):
            score = compare_histograms(first_hist, hist)
        log.info('Time for compare_histograms: %.3f', (time.time() - t))

        log.info('Score %0.3f',  (score, ))
        cv2.imshow('first', first_img)
        cv2.imshow('second', image)

        if cv2.waitKey(0) == 27:
            sys.exit(-1)
