import numpy as np
import cv2

# from structure_sfta import sfta
from utils import find_point_closest_to, find_most_distant_pair, find_random_points


def image_patches(img, patch_size=(64, 64), patch_step=5, max_count=5):
    has_mask = img.shape[2] == 4
    img_w = img.shape[1]
    img_h = img.shape[0]

    if not has_mask:
        raise Exception('Image has no mask')

    if img_w < patch_size[0] or img_h < patch_size[1]:
        return []

    a_img = img[:, :, 3]
    color_img = img[:, :, 0:3]

    _, a_thresh_img = cv2.threshold(a_img, 128, 255, cv2.THRESH_BINARY)

    box_w = patch_size[0]
    box_h = patch_size[1]

    box_a = box_w * box_h
    a_thresh = (box_a - 1)

    boxes = []

    for i in range(0, img_w - box_w, patch_step):
        for j in range(0, img_h - box_h, patch_step):
            roi = a_thresh_img[j:j + box_h, i:i + box_w]
            window_a = np.flatnonzero(roi).shape[0]

            if window_a > a_thresh:
                boxes.append((i, j))

    if len(boxes) == 0:
        best_l = [(img_w // 2 - box_w // 2, img_h // 2 - box_h // 2)]
    else:
        best_l = find_random_points(boxes, count=max(0, max_count - 3))
        center_box = find_point_closest_to(boxes, (img_w // 2 - box_w // 2, img_h // 2 - box_h // 2))   # Always works
        distant_boxes = find_most_distant_pair(boxes)

        if distant_boxes is not None:
            best_l.append(distant_boxes[0])
            best_l.append(distant_boxes[1])

        best_l.append(center_box)

    imgs = []
    for b in best_l:
        window_img = color_img[b[1]:b[1] + box_h, b[0]:b[0] + box_w, :].copy()
        imgs.append(window_img)

    return imgs
