#!/usr/bin/env python3
import cv2
import numpy as np

from log import makelog
log = makelog(__name__)


def histogram_lab(img, n_bins=16):
    """
    This feature extractor takes in a color image and returns a normalized color
    histogram of the pixel counts of each hue.
    """

    has_mask = img.shape[2] == 4

    color = img[:, :, 0:3] if has_mask else img
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)    # Convert to lab for better perceptual accuracy
    lab_flat = lab.reshape((color.shape[0] * color.shape[1], 3))

    if has_mask:
        mask = img[:, :, 3]
        mask = mask.reshape((mask.shape[0] * mask.shape[1], 1))
        lab_flat = lab_flat[np.where(mask > 128)[0]]

    l = lab_flat[:, 0]
    a = lab_flat[:, 1]
    b = lab_flat[:, 2]

    hist_l = np.histogram(l, n_bins, density=True, range=(0, 255))[0]
    hist_a = np.histogram(a, n_bins, density=True, range=(0, 255))[0]
    hist_b = np.histogram(b, n_bins, density=True, range=(0, 255))[0]

    hist = np.concatenate((hist_l, hist_a, hist_b), axis=0)
    return hist


def kmeans_quantization(img, n_clusters):
    """ Quantizises the image into a given number of clusters (n_clusters).
        This can work with images (4 channel) that have an alpha channel, this gets ignored,
        but it will "spend" one cluster for that
    """
    has_mask = img.shape[2] == 4

    color = img[:, :, 0:3] if has_mask else img
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)    # Convert to lab for better perceptual accuracy
    lab = lab.reshape((color.shape[0] * color.shape[1], 3))

    flab = np.float32(lab)

    if has_mask:
        mask = img[:, :, 3]
        mask = mask.reshape((mask.shape[0] * mask.shape[1], 1))
        flab[np.where(mask == 0)[0]] = (-255, -255, -255)  # Mask off alpha areas

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, labels, centers = cv2.kmeans(data=flab, K=n_clusters, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

    exclude_idx = np.where(centers[:, -1] < 0)[0]
    centers_u8 = np.uint8(centers)

    res = centers_u8[labels.flatten()]

    hist = centroid_histogram(labels)
    hist[exclude_idx] = 0.0

    dominant = centers_u8[np.argmax(hist)]

    return dominant, hist, centers_u8, res, labels


def centroid_histogram(labels):
    num_labels = np.arange(0, len(np.unique(labels)) + 1)
    (hist, _) = np.histogram(labels, bins=num_labels)

    hist = hist.astype('float')
    hist /= hist.sum()

    return hist


def find_dominant_colors(img):
    """Expects an image with or without alpha channel and
       returns the dominant colors (LAB) and their distribution (histogram)"""
    small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    dominant, hist, centers, processed, labels = kmeans_quantization(small_img, n_clusters=4)

    p = hist.argsort()[::-1][:-1]   # Sort and trim of background index (last)
    hist_sorted = hist[p]
    centers_sorted = centers[p]

    # return dominant, hist, centers, labels
    return centers_sorted, hist_sorted


def compare_colors(a, b):
    a = np.array(a, dtype=np.float)
    b = np.array(b, dtype=np.float)

    return np.linalg.norm(a - b)


def lab_to_rgb(c):
    dummy = np.array([np.array([c])])
    rgb = (cv2.cvtColor(dummy, cv2.COLOR_LAB2BGR)[0, 0]).tolist()

    return rgb


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

    for fn in pngfiles:
        log.info('Loading %s', fn)
        image = cv2.imread(os.path.join(p, fn), -1)
        (h, w) = image.shape[:2]

        t = time.time()

        centers, hist = find_dominant_colors(image)

        small_img = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        # dominant, hist, centers, processed, labels = kmeans_quantization(small_img, n_clusters=5)

        dominant = centers[0]
        rgb_dominant = lab_to_rgb(dominant)

        # print(len(centers))
        # print(len(hist))

        # num_clusts = len(centers)
        # for i in range(num_clusts):
        #     x = int(i / num_clusts * w)
        #     bh = int(h * hist[i] * 0.5)

        #     cv2.rectangle(image, (x, h - bh), (x + 20, h), lab_to_rgb(centers[i]), cv2.FILLED)

        # cv2.rectangle(image, (0, 0 , (x + 20, h), lab_to_rgb(centers[i]), cv2.FILLED)

        cv2.circle(image, (w // 2, h // 2), w // 8, rgb_dominant, -1)
        # log.info('Time taken: %.3f', (time.time() - t))

        limg = histogram_lab(image)

        cv2.imshow('image', image)
        if cv2.waitKey(0) == 27:
            sys.exit(-1)
