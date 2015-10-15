import cv2
import numpy as np

w, h = 400, 400

img = np.zeros((w, h), np.float)


while True:
    rad = 100.0   # max curvature influence     TODO: strength is missing...

    n = (1.0, 0.5)   # direction of curvature
    n = n / np.linalg.norm(n)
    
    pt = (100, 200)   # point of curvature

    xlin = np.arange(0, w)
    ylin = np.arange(0, h)

    xs, ys = np.meshgrid(xlin, ylin)

    coords = np.dstack((xs, ys))
    vecs = coords - pt
    mags = np.sqrt((vecs ** 2).sum(-1))[..., np.newaxis]

    vecs = vecs / mags

    dots = (vecs.dot(n) - 0.9) * 10.0  # TODO: adjust

    grad = 1.0 - np.clip(mags[:, :, 0] / rad, 0, 1.0)
    result = 1.0 - (grad * dots)

    cv2.imshow('dot products', dots * 1.0)
    cv2.imshow('mag gradient', grad * 1.0)
    cv2.imshow('result', result * 1.0)

    cv2.waitKey(1)
