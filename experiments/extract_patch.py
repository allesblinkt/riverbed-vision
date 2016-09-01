import numpy as np
import cv2
import random
import sys

sys.path.insert(0, '../brain')

from structure_sfta import sfta

img = cv2.imread('testdata/stone1.png', -1)

has_mask = img.shape[2] == 4

if not has_mask:
    print("No mask! Danger")

b, g, r, a = cv2.split(img)
col_img = cv2.merge((b, g, r))

_, thresh_img = cv2.threshold(a, 128, 255, cv2.THRESH_BINARY)

img_w = img.shape[1]
img_h = img.shape[0]
print(img_w, img_h)

box_w = 64
box_h = 64
box_a = box_w * box_h
a_thresh = box_a - 1

step = 5

boxes = []

for i in range(0, img_w - box_w, step):
    for j in range(0, img_h - box_h, step):
        roi = thresh_img[j:j + box_h, i:i + box_w]
        window_a = np.flatnonzero(roi).shape[0]

        if window_a > a_thresh:
            boxes.append((i, j))


def find_point_closest_to(points, point):
    best = None
    best_dsq = 0

    for p in points:
        d_x = point[0] - p[0]
        d_y = point[1] - p[1]
        dsq = d_x * d_x + d_y * d_y

        if best is None or dsq < best_dsq:
            best = p
            best_dsq = dsq

    return best


def find_most_distant_pair(points, iter=1000):
    best = None
    best_dsq = 0

    for i in range(iter):
        a = random.choice(points)
        b = random.choice(points)

        d_x = a[0] - b[0]
        d_y = a[1] - b[1]
        dsq = d_x * d_x + d_y * d_y

        if best is None or dsq > best_dsq:
            best = (a, b)
            best_dsq = dsq

    return best


def find_random(points, count=10):
    l = points.copy()
    random.shuffle(l)
    return l[:count]

center = find_point_closest_to(boxes, (img_w / 2 - box_w / 2, img_h / 2 - box_h / 2))
distant_pair = find_most_distant_pair(boxes)
randoms = find_random(boxes)

best_l = list(randoms)
best_l.append(distant_pair[0])
best_l.append(distant_pair[1])
best_l.append(center)

np.set_printoptions(precision=4, suppress=True)

draw_img = col_img.copy()

imgs = []
for best in best_l:
    lewindow = col_img[best[1]:best[1] + box_h, best[0]:best[0] + box_w, :]
    cv2.rectangle(draw_img, (best[0], best[1]), (best[0] + box_w, best[1] + box_h), (255, 0, 0))

    D, simgs = sfta(lewindow, 4)

    imgs.extend(simgs)


cv2.imshow("img", draw_img)

# print()
for img in imgs:
    nimg = np.array(img, dtype=np.uint8) * 255
    cv2.imshow("sfta", nimg)
    cv2.waitKey(0)

# # cv2.imshow("cl", cl1_img)
# cv2.imshow("window", lewindow)
# cv2.imshow("window_a", lewindow_a)
# cv2.imshow("window_b", lewindow_b)

cv2.waitKey(0)