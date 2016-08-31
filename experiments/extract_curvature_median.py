import cv2
import numpy as np

from random import randint, seed
import time

camera_port = -1
cam = None


def random_color():
    return (randint(0, 255), randint(0, 255), randint(0, 255))

if camera_port >= 0:
    cam = cv2.VideoCapture(camera_port)
    cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 360)
    cam.set(cv2.cv.CV_CAP_PROP_FPS, 30)
else:
    frame = cv2.imread('testdata/stones.jpg')


def falloff(x, x2, y, y2, pt, n, rad):   # max curvature influence     TODO: strength is missing...
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


while(True):

    seed(2)

    if cam:
        frame = cam.read()

    color_img = frame.copy()

    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    thresh_img = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 251, 2)


    # Cleaning
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations = 2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)


    # Contouring
    contours, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Curvature analysis of the external contours
    curvature_img = np.zeros_like(gray_img, dtype=np.float)

    contours_curvatures = []
    curvature_step = 7

    for contour in contours:
        c_len = len(contour)
        n = curvature_step

        concave_points = []

        #print "CCCCCC"

        if c_len <= n:
            continue

        for i in xrange(c_len):
            p_a = contour[i][0]  # prev
            p_b = contour[(i + n) % c_len][0]  # this
            p_c = contour[(i + n + n) % c_len][0]  # next

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

            curvature_img[p_b[1], p_b[0]] = angle * 0.5   # TODO: can go

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
                #print "Happening!"

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

            median_point = np.median(points, axis=0)
            mean_point = np.mean(points, axis=0)

            median_normal = np.median(normals, axis=0)
            mean_normal = np.mean(normals, axis=0)

            median_angle = np.median(cut_angles)
            mean_angle = np.mean(cut_angles)

            new_cuts.append((mean_point, median_point, mean_normal, median_normal, mean_angle, median_angle))

        contours_curvatures.append((contour, concave_points, cuts, new_cuts))

    # Draw concave normals
    cv2.drawContours(color_img, contours, -1, (255, 0, 0))


    t = time.time()

    normal_draw_length = 10.0
    for contour, contour_curvatures, cuts, new_cuts in contours_curvatures:
        # for idx, pt, normal, angle in contours_curvatures:
        #    n = np.uint16(np.round(pt + normal * normal_draw_length * angle))

        #    cv2.line(color_img, (pt[0], pt[1]), (n[0], n[1]), (0, 0, 255))

        # for cut in cuts:
        #     color = random_color()
        #     for idx, pt, normal, angle in cut:
        #         n = np.uint16(np.round(pt + normal * normal_draw_length * angle))

        #         cv2.line(color_img, (pt[0], pt[1]), (n[0], n[1]), color)

        for mean_point, median_point, mean_normal, median_normal, mean_angle, median_angle in new_cuts:
            color = random_color()

            p = median_point
            #print p
            n = np.uint16(np.round(p + median_normal * normal_draw_length * median_angle))
            #print n
            p = np.uint16(np.round(p))
            cv2.line(color_img, (p[0], p[1]), (n[0], n[1]), color)

    #thresh_img = np.ones()
    #print thresh_img


    thresh_img = thresh_img / 255.0
    print "Gffff"

    for contour, contour_curvatures, cuts, new_cuts in contours_curvatures:
        #print "frrooo"
        for mean_point, median_point, mean_normal, median_normal, mean_angle, median_angle in new_cuts:
            print np.linalg.norm(mean_normal)

            if np.linalg.norm(mean_normal) < 0.0001:
                print "Normal too short"
                continue

            #print "Normal good"

            rad = 35.0

            #print fo
            pt = median_point

            x = max(0, pt[0] - rad)
            y = max(0, pt[1] - rad)

            x2 = min(thresh_img.shape[1], pt[0] + rad)
            y2 = min(thresh_img.shape[0], pt[1] + rad)
            
            fo = falloff(x, x2, y, y2, pt, median_normal, rad)


            print thresh_img[y:y2, x:x2].shape
            print fo.shape
            thresh_img[y:y2, x:x2] *= fo


    print time.time() - t

    print thresh_img
        
    ttt = np.uint8(np.clip(thresh_img * 255.0, 0, 255))
    #print ttt
    _, ttt = cv2.threshold(ttt, 128, 255, cv2.THRESH_BINARY)
    print ttt.shape
    #ttt = cv2.adaptiveThreshold(ttt, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 251, 2)


    dist_transform = cv2.distanceTransform(ttt, cv2.cv.CV_DIST_L2, 5)
    print "MAAAAX", np.max(dist_transform)
    _, dist_thresh_img = cv2.threshold(dist_transform, np.max(dist_transform) * 0.2, 255, cv2.THRESH_BINARY)
    #dist_transform = cv2.normalize(dist_transform)

    markers = np.zeros((thresh_img.shape[0], thresh_img.shape[1]), dtype = np.int32)

    kernels, _ = cv2.findContours(np.uint8(dist_thresh_img.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for id in range(len(kernels)):
        cv2.drawContours(markers, kernels, id, id + 1, -1)

    unknown = cv2.subtract(sure_bg, np.uint8(dist_thresh_img))

    #_, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0      # mark the region of unknown with zero

    segmented = np.zeros_like(frame)

    cv2.watershed(segmented, markers)
    segmented[markers == -1] = (255, 0, 0)

    # TODO:
    # * Apply curvature weighting
    # * Find nuclei
    # * Watershed
    # * Label, extract
    # 
    _, areas = cv2.threshold(np.uint8(markers), 1, 255, cv2.THRESH_BINARY)
    areas[markers == -1] = 0
    stones, _ = cv2.findContours(areas.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    print stones
    for id in range(len(stones)):
        c = random_color()
        cv2.drawContours(color_img, stones, id, c, 1)
       

    print "Marker type!", markers
    # cv2.imshow('curvature_map', thresh_img / 255.0 * 0.1)
    cv2.imshow('threshold', ttt)
    cv2.imshow('dist', segmented)
    cv2.imshow('debug', areas)


    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# cam.release()
cv2.destroyAllWindows()