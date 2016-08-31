import cv2
import numpy as np

from random import randint, seed

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

while(True):

    seed(2)

    if cam:
        frame = cam.read()

    color_img = frame.copy()

    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    thresh_img = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 251, 2)

    # Contouring
    contours, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Curvature analysis of the external contours
    curvature_img = np.zeros_like(gray_img, dtype=np.float)

    contours_curvatures = []
    curvature_step = 7

    for contour in contours:
        c_len = len(contour)
        n = curvature_step

        concave_points = []

        print "CCCCCC"

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

        cuts = [(cut_points, cut_normals)]
        last_idx = -10    # TODO: wrap around or merge later...
        for point in concave_points:
            idx, pt, normal, angle = point

            if idx - last_idx > 5:  # TODO: make threshold adjustable
                cut_points = []
                cut_normals = []
                cuts.append((cut_points, cut_normals))
                print "Happening!"

            cut_points.append(pt)
            cut_normals.append(normal)
            last_idx = idx

        new_cuts = []
        for cut_points, cut_normals in cuts:
            if len(cut_points) < 2:
                continue

            points = np.array(cut_points)
            normals = np.array(cut_normals)

            median_point = np.median(points, axis=0)
            mean_point = np.mean(points, axis=0)

            median_normal = np.median(normals, axis=0)
            mean_normal = np.mean(normals, axis=0)

            new_cuts.append((mean_point, median_point, mean_normal, median_normal))

        contours_curvatures.append((contour, concave_points, cuts, new_cuts))

    # Draw concave normals
    cv2.drawContours(color_img, contours, -1, (255, 0, 0))

    normal_draw_length = 10.0
    for contour, contours_curvatures, cuts, new_cuts in contours_curvatures:
        # for idx, pt, normal, angle in contours_curvatures:
        #    n = np.uint16(np.round(pt + normal * normal_draw_length * angle))

        #    cv2.line(color_img, (pt[0], pt[1]), (n[0], n[1]), (0, 0, 255))

        # for cut in cuts:
        #     color = random_color()
        #     for idx, pt, normal, angle in cut:
        #         n = np.uint16(np.round(pt + normal * normal_draw_length * angle))

        #         cv2.line(color_img, (pt[0], pt[1]), (n[0], n[1]), color)

        for mean_point, median_point, mean_normal, median_normal in new_cuts:
            color = random_color()

            p = median_point
            print p
            n = np.uint16(np.round(p + median_normal * normal_draw_length))
            print n
            p = np.uint16(np.round(p))
            cv2.line(color_img, (p[0], p[1]), (n[0], n[1]), color)


    # TODO:
    # * Apply curvature weighting
    # * Find nuclei
    # * Watershed
    # * Label, extract

    cv2.imshow('curvature_map', curvature_img)
    cv2.imshow('debug', color_img)
    # cv2.imshow('threshold', thresh_img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# cam.release()
cv2.destroyAllWindows()