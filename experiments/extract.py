import cv2
import numpy as np
from random import randint, uniform, seed
import time

camera_port = -1
cam = None
shaky_fake = False
save_stones = False

if camera_port >= 0:
    cam = cv2.VideoCapture(camera_port)
    cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 360)
    cam.set(cv2.cv.CV_CAP_PROP_FPS, 30)
else:
    fakecam = cv2.imread('testdata/stones.jpg')


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


def process_stone(id, stones_contours, src_img, result_img):
    contour = stones_contours[id]
    m = cv2.moments(contour)

    try:
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])
    except:
        return


    bbox = cv2.boundingRect(contour)

    ec, es, ea = cv2.minAreaRect(contour)
    ec = (int(ec[0]), int(ec[1]))
    es = (int(es[0]) / 2, int(es[1]) / 2)
    ea = int(ea)

    cutout = src_img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0]+ bbox[2]]
    b, g, r = cv2.split(cutout)
    a = np.zeros_like(b, dtype=np.uint8)
    cv2.drawContours(a, stones_contours, id, 255, -1, offset=(-bbox[0], -bbox[1]))
    cropped = cv2.merge((b,g,r,a))
    avg, dev = cv2.meanStdDev(cutout, mask=a)
    avg, dev = avg.T[0], np.max(dev.T[0])

    if save_stones:
        cv2.imwrite('stone_{:03d}.png'.format(id), cropped)

    color = np.multiply(avg, 2)
    cv2.drawContours(result_img, stones_contours, id, color, -1)
    cv2.circle(result_img, (cx, cy), 4, (0, 128, 0))
    cv2.rectangle(result_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0))
    cv2.ellipse(result_img, ec, es, ea, 0, 360, (0, 0, 255))

    print 'id:{} bbox:{} ecenter:{} esize:{} eangle:{} avg:{} dev:{}'.format(id, bbox, ec, es, ea, avg, dev)

while(True):

    if cam:
        frame = cam.read()
    else:
        if not shaky_fake:
            frame = fakecam.copy()
        else:
            # rotate and zoom randomly input frame
            rows, cols, _ = fakecam.shape
            x, y = uniform(-10, 10), uniform(-10, 10)
            M = np.float32(((1,0,x), (0,1,y)))
            frame = cv2.warpAffine(fakecam, M, (cols, rows))

    start_time = time.time()

    color_img = frame.copy()

    # Grayscale conversion, blurring, threshold
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    thresh_img = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 251, 2)

    # Cleaning
    kernel = np.ones((3, 3), np.uint8)
    opening_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg_img = cv2.dilate(opening_img, kernel, iterations=3)

    # Contouring
    contours, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(color_img, contours, -1, (255, 0, 0))

    # Curvature analysis of the external contours
    curvature_img = np.zeros_like(gray_img, dtype=np.float)

    # Create and multiply the weighting image
    weight_img = thresh_img.copy() / 255.0

    for contour in contours:
        cuts = analyze_contour_cuts(contour)

        if not cuts:
            continue

        for cut_point, cut_normal, cut_angle in cuts:
            draw_normal(color_img, cut_point, cut_normal, cut_angle)

            if np.linalg.norm(cut_normal) < 0.0001:  # Normal too short
                continue

            rad = 35.0   # FIXME: roi index calculation

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
    _, dist_thresh_img = cv2.threshold(dist_transform, np.max(dist_transform) * 0.2, 255, cv2.THRESH_BINARY)

    markers_img = np.zeros((thresh_img.shape[0], thresh_img.shape[1]), dtype=np.int32)

    kernel_contours, _ = cv2.findContours(np.uint8(dist_thresh_img.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for id in range(len(kernel_contours)):
        cv2.drawContours(markers_img, kernel_contours, id, id + 1, -1)

    unknown_img = cv2.subtract(sure_bg_img, np.uint8(dist_thresh_img))

    markers_img = markers_img + 1
    markers_img[unknown_img == 255] = 0      # mark the region of unknown with zero

    segmented_img = markers_img.copy()
    cv2.watershed(color_img, segmented_img)
    segmented_img[segmented_img == 1] = -1

    # Find individual stones and draw them
    stones_contours, _ = cv2.findContours(segmented_img, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_NONE)

    result_img = np.zeros_like(color_img)

    for id in range(len(stones_contours)):
        process_stone(id, stones_contours, frame, result_img)

    elapsed_time = time.time() - start_time
    print 'Analysis took: {:0.3f}s'.format(elapsed_time)

    # Display
    cv2.imshow('color with debug', color_img)
    cv2.imshow('curvature weighting', weight_img)
    cv2.imshow('curvature weighting threshold', weight_thresh_img)
    cv2.imshow('markers', markers_img * 256)
    cv2.imshow('stones', result_img)

    # Input
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# cam.release()
cv2.destroyAllWindows()
