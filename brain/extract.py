#!/usr/bin/python
import cv2
import numpy as np
import time
import sys
from multiprocessing import Pool

from coloranalysis import find_dominant_colors, lab_to_rgb
from structure import lbp_histogram
from stone import Stone

from log import makelog
log = makelog(__name__)

p_scalef = 4

blank_img = cv2.imread('blank.png')
blank_small_img = cv2.resize(blank_img, (blank_img.shape[1]//p_scalef, blank_img.shape[0]//p_scalef))


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


def preselect_stone(img_size, center, dim, bbox):
    min_center_border_dist = 100
    min_axis_size = 50
    bbox_tolerance = 4

    # center too close to the edge
    if center[0] < min_center_border_dist or center[0] > img_size[0] - min_center_border_dist:
        return False
    if center[1] < min_center_border_dist or center[1] > img_size[1] - min_center_border_dist:
        return False

    # Touches edges
    if bbox[0] < bbox_tolerance or bbox[0] + bbox[2] > img_size[0] - bbox_tolerance -1:
        return False   # touches X edges
    if bbox[1] < bbox_tolerance or bbox[1] + bbox[3] > img_size[1] - bbox_tolerance -1:
        return False   # touches Y edges

    # too small (either axis)
    if dim[0] < min_axis_size and dim[1] < min_axis_size:
        return False

    small_img_side = min(img_size[0], img_size[1])
    max_axis = small_img_side * 0.90 * 0.5

    # too big, machine sees carriage
    if dim[0] > max_axis or dim[1] > max_axis:
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
    
    fit_center = (int(ec[0]), int(ec[1]))
    fit_dim = (int(es[0]) // 2, int(es[1]) // 2)
    fit_angle = int(ea)
    if fit_dim[1] > fit_dim[0]:
        fit_dim = fit_dim[1], fit_dim[0]
        fit_angle += 90
    fit_angle = fit_angle % 180

    img_h, img_w, _ = src_img.shape

    if not preselect_stone((img_w, img_h), fit_center, fit_dim, bbox):
        return None

    cutout = src_img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0]+ bbox[2]]
    b, g, r = cv2.split(cutout)
    a = np.zeros_like(b, dtype=np.uint8)
    cv2.drawContours(a, [contour], 0, 255, -1, offset=(-bbox[0], -bbox[1]))
    cropped = cv2.merge((b, g, r, a))

    colors_lab, colors_hist = find_dominant_colors(cropped)
    dominant_color = colors_lab[0]
    structure = lbp_histogram(cropped)

    if result_img is not None:
        rgb_color = lab_to_rgb(dominant_color)

        cv2.drawContours(result_img, [contour], 0, rgb_color, -1)
        cv2.circle(result_img, fit_center, 4, (128, 0, 0))
        cv2.rectangle(result_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0))

        cv2.ellipse(result_img, fit_center, fit_dim, fit_angle, 0, 360, (0, 0, 255))

    stone_data = Stone(fit_center, fit_dim, fit_angle, dominant_color, structure)

    if save_stones:
        cv2.imwrite('stones/stone_{}_{:03d}.{}'.format(frame_desc, id, save_stones), cropped)
        stone_data.save('stones/stone_{}_{:03d}.data'.format(frame_desc, id))

    return stone_data


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

    # "OR" with tight thresshold of values
    thresh_s_img[thresh_v_sure_img > 128] = 255

    thresh_img = thresh_s_img



    # thresh_img = cv2.resize(thresh_img, (thresh_img.shape[1]*4, thresh_img.shape[0]*4), interpolation=cv2.INTER_CUBIC)
    thresh_img = cv2.GaussianBlur(thresh_img, (9, 9), 0)   
    _, thresh_img = cv2.threshold(thresh_img, 128, 255, cv2.THRESH_BINARY)

    return thresh_img


def process_image(frame_desc, color_img, save_stones=None, debug_draw=False, debug_wait=False):

    log.debug('Start processing image: %s', frame_desc)
    start_time = time.time()
    
    small_img = cv2.resize(color_img, (color_img.shape[1]//p_scalef, color_img.shape[0]//p_scalef))

    # subtract blank vignette
    small_img = 255 - cv2.subtract(blank_small_img, small_img)
    thresh_img = threshold_adaptive_with_saturation(small_img)
    
    # Cleaning
    kernel = np.ones((3, 3), np.uint8)
    opening_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg_img = cv2.dilate(opening_img, kernel, iterations=3)

    # Contouring
    _, contours, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Curvature analysis of the external contours
    curvature_img = np.zeros_like(thresh_img, dtype=np.float)

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

            x1 = int(round(max(0, cut_point[0] - rad)))
            y1 = int(round(max(0, cut_point[1] - rad)))

            x2 = int(round(min(thresh_img.shape[1], cut_point[0] + rad)))
            y2 = int(round(min(thresh_img.shape[0], cut_point[1] + rad)))

            falloff_part = falloff_gradient(x1, x2, y1, y2, cut_point, cut_normal, rad)

            try:
                weight_img[y1:y2, x1:x2] *= falloff_part
            except:
                pass

    # Threshold the weighting image
    weight_thresh_img = np.uint8(np.clip(weight_img * 255.0, 0, 255))
    _, weight_thresh_img = cv2.threshold(weight_thresh_img, 128, 255, cv2.THRESH_BINARY)

    # Standard distance transform and watershed segmentation
    dist_transform = cv2.distanceTransform(weight_thresh_img, cv2.DIST_L2, 5)
    _, dist_thresh_img = cv2.threshold(dist_transform, 10, 255, cv2.THRESH_BINARY)   # Constant distance...

    markers_img = np.zeros((thresh_img.shape[0], thresh_img.shape[1]), dtype=np.int32)

    dist_thresh_img_u8 = np.uint8(dist_thresh_img)
    _, kernel_contours, _ = cv2.findContours(dist_thresh_img_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for id in range(len(kernel_contours)):
        cv2.drawContours(markers_img, kernel_contours, id, id + 1, -1)

    unknown_img = cv2.subtract(sure_bg_img, dist_thresh_img_u8)

    markers_img = markers_img + 1
    markers_img[unknown_img == 255] = 0      # mark the region of unknown with zero

    segmented_img = markers_img.copy()

    #watershed_img = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2RGB)
    watershed_img = small_img
    cv2.watershed(watershed_img, segmented_img)
    segmented_img[segmented_img == 1] = -1

    result_img = np.zeros_like(color_img)

    # Find individual stones and analyze them
    _, stones_contours, _ = cv2.findContours(segmented_img, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_NONE)

    pool = Pool(8)    # TODO: make the number of processes configurable

    contours_and_args = []
    for id, contour in enumerate(stones_contours):
        contour *= p_scalef   # enlarge to compensate for smaller processed image
        # contours_and_args.append([frame_desc, id, contour, color_img, result_img, save_stones])
        contours_and_args.append([frame_desc, id, contour, color_img, None, save_stones])

    stones = pool.starmap(process_stone, contours_and_args)

    pool.close()

    # Keep stones with a result
    stones = [stone for stone in stones if stone is not None]


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
        
        if debug_wait:
            key = cv2.waitKey()
        else:
            key = cv2.waitKey(1)

        if key == ord('q'):
            sys.exit(1)

    return stones, result_img, thresh_img, weight_img


def main():
    global blank_img
    global blank_small_img

    import os
    import fnmatch

    jpgfiles = []
    p = 'map_offline'

    for file in os.listdir('map_offline'):
        if fnmatch.fnmatch(file, 'grab_*.jpg'):
            jpgfiles.append(file)

    # jpgfiles = ["grab_0429_1380.jpg"]

    for fn in jpgfiles:
        full_fn = os.path.join(p, fn)
        log.info('Processing %s', full_fn)
        frame = cv2.imread(full_fn)
        stones, result_img, thresh_img, weight_img = process_image(fn, frame, save_stones='png', debug_draw=True, debug_wait=False)



    # for i in range(13, 30+1):
    #     frame = cv2.imread('../experiments/testdata/photo-{}.jpg'.format(i))
    #     process_image('photo-{}'.format(i), frame, save_stones='png', debug_draw=False)

if __name__ == "__main__":
    main()
