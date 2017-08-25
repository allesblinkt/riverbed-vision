#!/usr/bin/env python3

import cv2
import numpy as np

from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq

from log import makelog
log = makelog(__name__)

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

    try:
        score = cv2.compareHist(arr_a, arr_b, cv2.HISTCMP_CHISQR)
    except:
        log.warn('Could not compare histograms.')
        score = 1.0

    return score


def plot_hist(hist):
    graph_w = 300
    graph_h = 200
    graph_img = np.zeros((graph_h, graph_w,3), np.uint8)

    num_bins = len(hist)
    for i in range(1, num_bins-2, 1):
        x = int(i / num_bins * graph_w)
        bh = int(graph_h * hist[i] * 10.0)
        bw = int(graph_w / num_bins)
        cv2.rectangle(graph_img, (x, graph_h-bh), (x+bw, graph_h), [255, 0, 0], cv2.FILLED)

    return graph_img

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

    pngfiles = ["a.png", "b.png", "a.png", "b.png", "a.png", "b.png", "a.png", "b.png", "a.png", "b.png", "a.png", "b.png", "a.png", "b.png", "a.png", "b.png"]

    first_path = os.path.join(p, pngfiles[0])
    first_img = cv2.imread(first_path, -1)

    first_hist = lbp_histogram(first_img)

    for fn in pngfiles:
        log.info('Loading %s', fn)
        image = cv2.imread(os.path.join(p, fn), -1)
        (h, w) = image.shape[:2]

        t = time.time()
        hist_a = lbp_histogram(image, radius=3)
        hist_b = lbp_histogram(image, radius=12)
        #print(hist)
        log.info('Time for lbp_histogram: %.3fs', (time.time() - t))

        # t = time.time()
        # for i in range(1000):
        #     score = compare_histograms(first_hist, hist)
        # print(score)
        # log.info('Time for compare_histograms: %.3fs', (time.time() - t))

        # import matplotlib.pyplot as plt
        # plt.plot(hist)
        # plt.ylabel('some numbers')
        # plt.show()

        graph_a_img = plot_hist(hist_a)
        graph_b_img = plot_hist(hist_b)


        # log.info('Score %0.3f',  (score, ))
        cv2.imshow('first', first_img)
        cv2.imshow('second', image)
        cv2.imshow('histogram_a', graph_a_img)
        cv2.imshow('histogram_b', graph_b_img)

        if cv2.waitKey(0) == 27:
            sys.exit(-1)
