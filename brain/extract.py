#!/usr/bin/python
import cv2
import numpy as np
import time
import sys

from coloranalysis import find_dominant_color, lab_to_rgb
from structure import lbp_histogram
from stone import Stone

from log import makelog
log = makelog(__name__)

blank_img = cv2.imread('blank.png')
blank_half_img = cv2.resize(blank_img, (blank_img.shape[1]//2, blank_img.shape[0]//2))


def analyze_contour_cuts(contour, step=7):
    """
    Parameters
    ----------
    contour: array_like
        The contour to be analyzed

    Returns
    -------

    None if the contour is invalid (i.e. too little points to analyze)
    """

    c_len = len(contour)

    concave_points = []

    if c_len <= step:
        return None

    try: # xrange is range in Python 3
        xrange
    except NameError:
        xrange = range

    for i in xrange(c_len):
        p_a = contour[i][0]  # prev
        p_b = contour[(i + step) % c_len][0]  # this
        p_c = contour[(i + 2 * step) % c_len][0]  # next

        v_a = p_b - p_a
        v_b = p_b - p_c

        mag_a = np.linalg.norm(v_a)
        mag_b = np.linalg.norm(v_b)
        dot = mag_a * mag_b

        angle_cos = (v_a[0] * v_b[0] + v_a[1] * v_b[1]) / dot
        angle_cos = max(-1.0, min(angle_cos, 1.0))  # clamp for float inaccuracy
        angle = np.arccos(angle_cos)

        if angle_cos < 0:
            angle = np.pi - angle

        cross = np.cross(v_a, v_b)
        angle *= -np.sign(cross)  # Add heading to curvature depending on winding

        if (angle > 0.5):    # TODO: make threshold adjustable
            normal = v_a / mag_a + v_b / mag_b
            normal = normal / np.linalg.norm(normal)
            concave_points.append([i, p_b, normal, angle])

    # Group concave spots into cuts
    cut_points = []
    cut_normals = []
    cut_angles = []

    cuts = [(cut_points, cut_normals, cut_angles)]
    last_idx = -10    # TODO: wrap around or merge later...
    for point in concave_points:
        idx, pt, normal, angle = point

        if idx - last_idx > 5:  # TODO: make threshold adjustable
            cut_points = []
            cut_normals = []
            cut_angles = []
            cuts.append((cut_points, cut_normals, cut_angles))

        cut_points.append(pt)
        cut_normals.append(normal)
        cut_angles.append(angle)
        last_idx = idx

    new_cuts = []
    for cut_points, cut_normals, cut_angles in cuts:
        if len(cut_points) < 2:
            continue

        points = np.array(cut_points)
        normals = np.array(cut_normals)
        angles = np.array(cut_angles)

        point = np.median(points, axis=0)
        normal = np.median(normals, axis=0)
        angle = np.median(angles, axis=0)

        new_cuts.append((point, normal, angle))

    return new_cuts


def falloff_gradient(x, x2, y, y2, pt, n, rad):   # max curvature influence     TODO: strength is missing...
    """
    Calculates the falloff gradient image for a given normal

    Parameters
    ----------
    """

    n = (n[1], n[0]) / np.linalg.norm(n)

    xlin = np.arange(x, x2)
    ylin = np.arange(y, y2)

    xs, ys = np.meshgrid(xlin, ylin)

    coords = np.dstack((ys, xs))
    vecs = coords - (pt[1], pt[0])
    mags = np.sqrt((vecs ** 2).sum(-1))[..., np.newaxis]

    mags = np.clip(mags, 0.0001, 100000.0)

    vecs = vecs / mags

    dots = (vecs.dot(n) - 0.5) * 2.0  # TODO: adjust

    grad = 1.0 - np.clip(mags[:, :, 0] / rad, 0, 1.0)
    result = 1.0 - (grad * dots)

    return result


def draw_normal(img, pt, normal, angle, scale=10.0):
    """ Draws the normal onto an image at a given point """
    start = np.uint16(np.round(pt))
    end = np.uint16(np.round(pt + normal * scale * angle))
    cv2.line(img, (start[0], start[1]), (end[0], end[1]), (0, 0, 255))


def preselect_stone(shape, ec, es):
    # too close to the edge
    if ec[0] < 100 or ec[0] > shape[0] - 100:
        return False
    if ec[1] < 100 or ec[1] > shape[1] - 100:
        return False

    # too small
    if es[0] < 50 and es[1] < 50:
        return False

    small_side = min(shape[0], shape[1])
    max_axis = small_side * 0.90 * 0.5

    # too big, machine sees carriage
    if es[0] > max_axis or es[1] > max_axis:
        return False

    return True


def process_stone(frame_desc, id, contour, src_img, result_img, save_stones=None):
    m = cv2.moments(contour)

    try:
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])
    except:
        return None

    bbox = cv2.boundingRect(contour)

    ec, es, ea = cv2.minAreaRect(contour)
    ec = (int(ec[0]), int(ec[1]))
    es = (int(es[0]) / 2, int(es[1]) / 2)
    ea = int(ea)
    if es[1] > es[0]:
        es = es[1], es[0]
        ea += 90
    ea = ea % 180

    resy, resx, _ = src_img.shape
    if not preselect_stone((resx, resy), ec, es):
        return None

    cutout = src_img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0]+ bbox[2]]
    b, g, r = cv2.split(cutout)
    a = np.zeros_like(b, dtype=np.uint8)
    cv2.drawContours(a, [contour], 0, 255, -1, offset=(-bbox[0], -bbox[1]))
    cropped = cv2.merge((b, g, r, a))
    color = find_dominant_color(cropped)
    structure = lbp_histogram(cropped)

    if result_img is not None:
        rgb_color = lab_to_rgb(dominant_color)
        # dummy = np.array([np.array([color])])
        # rgb_color = (cv2.cvtColor(dummy, cv2.COLOR_LAB2BGR)[0, 0]).tolist()
        cv2.drawContours(result_img, [contour], 0, rgb_color, -1)
        cv2.circle(result_img, ec, 4, (128, 0, 0))
        cv2.rectangle(result_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0))
        cv2.ellipse(result_img, ec, es, ea, 0, 360, (0, 0, 255))

    ret = Stone(ec, es, ea, color, structure)

    if save_stones:
        cv2.imwrite('stones/stone_{}_{:03d}.{}'.format(frame_desc, id, save_stones), cropped)
        ret.save('stones/stone_{}_{:03d}.data'.format(frame_desc, id))

    return ret


def threshold_adaptive_with_saturation(image):
    """ Expects an RGB image and thresholds based on saturation and value channels."""
   
    # image = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))

    # Grayscale conversion, blurring, threshold
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_img, s_img, v_img = cv2.split(hsv_img)
       
    gray_s_img = cv2.GaussianBlur(255-s_img, (15, 15), 0)
    gray_v_img = cv2.GaussianBlur(v_img, (5, 5), 0)

    thresh_v_img = cv2.adaptiveThreshold(gray_v_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 151, -30)
    thresh_s_img = cv2.adaptiveThreshold(gray_s_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 151, -15)

    thresh_v_sure_img = cv2.adaptiveThreshold(gray_v_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 151, 5)
    thresh_v_sure_img[gray_v_img > 252] = 0    # prevent adaptive runaway

    # Secondary static threshhold on saturation
    thresh_s_img[gray_s_img > 240] = 0

    # "AND" with relaxed thresshold of values
    thresh_s_img[thresh_v_img < 128] = 0

    # "OR" with tight thresshold of valus
    thresh_s_img[thresh_v_sure_img > 128] = 255

    thresh_img = thresh_s_img



    # thresh_img = cv2.resize(thresh_img, (thresh_img.shape[1]*4, thresh_img.shape[0]*4), interpolation=cv2.INTER_CUBIC)
    thresh_img = cv2.GaussianBlur(thresh_img, (9, 9), 0)   
    _, thresh_img = cv2.threshold(thresh_img, 128, 255, cv2.THRESH_BINARY)

    return thresh_img


def process_image(frame_desc, color_img, save_stones=None, debug_draw=False):

    log.debug('Start processing image: %s', frame_desc)
    start_time = time.time()
    
    half_img = cv2.resize(color_img, (color_img.shape[1]//2, color_img.shape[0]//2))

    # subtract blank vignette
    half_img = 255 - cv2.subtract(blank_half_img, half_img)
    thresh_img = threshold_adaptive_with_saturation(half_img)
    
    # Cleaning
    kernel = np.ones((3, 3), np.uint8)
    opening_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg_img = cv2.dilate(opening_img, kernel, iterations=3)

    # Contouring
    contours, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Curvature analysis of the external contours
    curvature_img = np.zeros_like(gray_img, dtype=np.float)

    # Create and multiply the weighting image
    weight_img = thresh_img.copy()

    # Draw the outer contours for the distance map
    for id in range(len(contours)):
        cv2.drawContours(weight_img, contours, id, 255, -1)

    weight_img = weight_img / 255.0

    for contour in contours:
        cuts = analyze_contour_cuts(contour)

        if not cuts:
            continue

        for cut_point, cut_normal, cut_angle in cuts:

            if np.linalg.norm(cut_normal) < 0.0001:  # Normal too short
                continue

            rad = 60.0

            x = max(0, cut_point[0] - rad)
            y = max(0, cut_point[1] - rad)

            x2 = min(thresh_img.shape[1], cut_point[0] + rad)
            y2 = min(thresh_img.shape[0], cut_point[1] + rad)

            falloff_part = falloff_gradient(x, x2, y, y2, cut_point, cut_normal, rad)

            try:
                weight_img[y:y2, x:x2] *= falloff_part
            except:
                pass

    # Threshold the weighting image
    weight_thresh_img = np.uint8(np.clip(weight_img * 255.0, 0, 255))
    _, weight_thresh_img = cv2.threshold(weight_thresh_img, 128, 255, cv2.THRESH_BINARY)

    # Standard distance transform and watershed segmentation
    dist_transform = cv2.distanceTransform(weight_thresh_img, cv2.cv.CV_DIST_L2, 5)
    _, dist_thresh_img = cv2.threshold(dist_transform, 10, 255, cv2.THRESH_BINARY)   # Constant distance...

    markers_img = np.zeros((thresh_img.shape[0], thresh_img.shape[1]), dtype=np.int32)

    dist_thresh_img_u8 = np.uint8(dist_thresh_img)
    kernel_contours, _ = cv2.findContours(dist_thresh_img_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for id in range(len(kernel_contours)):
        cv2.drawContours(markers_img, kernel_contours, id, id + 1, -1)

    unknown_img = cv2.subtract(sure_bg_img, dist_thresh_img_u8)

    markers_img = markers_img + 1
    markers_img[unknown_img == 255] = 0      # mark the region of unknown with zero

    segmented_img = markers_img.copy()
    cv2.watershed(half_img, segmented_img)
    segmented_img[segmented_img == 1] = -1

    # Find individual stones and draw them
    stones_contours, _ = cv2.findContours(segmented_img, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_NONE)

    result_img = np.zeros_like(color_img)

    stones = []
    for id, contour in enumerate(stones_contours):
        contour *= 2
        s = process_stone(frame_desc, id, contour, color_img, result_img, save_stones=save_stones)
        if s:
            stones.append(s)

    elapsed_time = time.time() - start_time

    log.debug('End processing image: %s, Analysis took: %0.3fs', frame_desc, elapsed_time)

    if debug_draw:
        cv2.imshow('color with debug', gray_img)
        cv2.imshow('threshold', thresh_img)
        cv2.imshow('distance threshold', dist_thresh_img)
        cv2.imshow('curvature weighting', weight_img)
        cv2.imshow('curvature weighting threshold', weight_thresh_img)
        cv2.imshow('markers', markers_img * 256)
        cv2.imshow('stones', result_img)
        key = cv2.waitKey()
        if key == ord('q'):
            sys.exit(1)

    return stones, result_img, thresh_img, weight_img


def main():
    global blank_img
    global blank_half_img

    frame = cv2.imread('map_offline/grab_2535_0897.jpg')
    stones, result_img, thresh_img, weight_img = process_image('bla', frame, save_stones='png', debug_draw=True)

    # for i in range(13, 30+1):
    #     frame = cv2.imread('../experiments/testdata/photo-{}.jpg'.format(i))
    #     process_image('photo-{}'.format(i), frame, save_stones='png', debug_draw=False)

if __name__ == "__main__":
    main()
