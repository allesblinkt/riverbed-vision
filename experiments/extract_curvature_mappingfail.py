#pylint: disable=no-name-in-module
import cv2
import numpy as np
import time

camera_port = -1
cam = None

if camera_port >= 0:
    cam = cv2.VideoCapture(camera_port)
    cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 360)
    cam.set(cv2.cv.CV_CAP_PROP_FPS, 30)
else:
    frame = cv2.imread('testdata/stones.jpg')


def find_biggest_contour(contours):
    biggest = None
    area = 0

    for contour in contours:
        if cv2.contourArea(contour) > area:
            biggest = contour
            area = cv2.contourArea(contour)

    return biggest


while(True):
    if cam:
        frame = cam.read()

    color_img = frame.copy()

    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    thresh_img = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 251, 2)

    # Contouring
    contours, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Curvature analysis of the external contours
    curvature_img = np.zeros_like(color_img, dtype=np.float)
    curvature_mag_img = np.zeros_like(gray_img, dtype=np.float32)

    concave_points = []
    curvature_step = 7

    # contours = [find_biggest_contour(contours)]

    for contour in contours:
        c_len = len(contour)
        n = curvature_step

        if c_len <= n:
            continue

        convex_hull = cv2.convexHull(contour, returnPoints = False)
        #print convex_hull
        defects = cv2.convexityDefects(contour, convex_hull)

        if defects is not None:
            for defect in defects:
                d = defect[0]
                start = contour[d[0]][0]
                end = contour[d[1]][0]

                print start

                cv2.line(color_img, (start[0], start[1]), (end[0], end[1]), (0, 0, 255))



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

            if not np.isnan(angle):  #angle > 0:
                normal = v_a / mag_a + v_b / mag_b
                normal = normal / np.linalg.norm(normal)

                if not np.isnan(normal[0]):
                    concave_points.append([p_b, normal, angle])

                    curvature_img[p_b[1], p_b[0]][0] = normal[0] * abs(angle)   # TODO: can go
                    curvature_img[p_b[1], p_b[0]][1] = normal[1] * abs(angle)   # TODO: can go
                    curvature_img[p_b[1], p_b[0]][2] = 0   # TODO: can go

                if angle > 0:
                    curvature_mag_img[p_b[1], p_b[0]] = abs(angle)   # TODO: can go


    kernel = np.ones((3, 3), np.uint8)

    # Draw concave normals
    cv2.drawContours(color_img, contours, -1, (255, 0, 0))
    _, curvature_mag_img = cv2.threshold(curvature_mag_img, 0.2 * curvature_mag_img.max(), 255, cv2.THRESH_BINARY)
    curvature_mag_img = np.uint8(255-curvature_mag_img)

    curvature_mag_img = cv2.erode(curvature_mag_img, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(curvature_mag_img, cv2.cv.CV_DIST_L2, 5)

    dist_transform =  cv2.normalize(dist_transform)
    #curvature_mag_img = np.uint8(cv2.normalize(curvature_mag_img) * 255.0)


    normal_draw_length = 40.0
    for pt, normal, angle in concave_points:
        n = np.uint16(np.round(pt + normal * normal_draw_length * angle))

        #cv2.line(color_img, (pt[0], pt[1]), (n[0], n[1]), (0, 0, 255))

    #t = int((time.time() * 10000.0) % 255.0)

    # print curvature_img[:][:][:].shape

    # for i in range(50):
    #     curvature_img = cv2.blur(curvature_img, (3, 3), 0)
    #     curvature_mag_img = cv2.blur(curvature_mag_img, (3, 3), 0)

    # im = curvature_mag_img

    # curvature_mag_img = cv2.normalize(curvature_mag_img)
    # curvature_mag_stack_img = np.asarray(np.dstack((im, im, im)), dtype=np.float)

    # curvature_img = np.multiply(curvature_mag_stack_img, curvature_img)
    # curvature_img = cv2.normalize(curvature_img)


    # for y in xrange(0, len(curvature_img), 5):
    #     for x in xrange(0, len(curvature_img[0]), 5):
    #         rn = curvature_img[y][x]
    #         pt = np.array([x, y, 0])

    #         n = np.uint16(np.round(pt + (rn * 400.0)))

    #         cv2.line(color_img, (pt[0], pt[1]), (n[0], n[1]), (0, 0, 255))













    # print len(concave_points)
    
    # def func(v, r, c):
    #     px = np.array([c, r])

    #     prod = 1.0

    #     for cp in concave_points:
    #         pt = cp[0]

    #         #if abs(pt[0] - px[0]) + abs(pt[1] - px[1]) < 10:
    #         #    prod += 100

    #         prod += 1  #np.linalg.norm(pt - px)


    #     return prod

    # vfunc = np.vectorize(func, otypes=[np.uint8])

    # mask = np.zeros_like(blur_img, dtype=np.uint8)
    # curvature_gradient = np.zeros_like(blur_img, dtype=np.uint8)

    #cv2.drawContours(mask, contours, 0, 255, -1)

    #curvature_gradient[mask.nonzero()] = vfunc(curvature_gradient[mask.nonzero()], mask.nonzero()[0], mask.nonzero()[1])

    # TODO:
    # * Apply curvature weighting
    # * Find nuclei
    # * Watershed
    # * Label, extract


    cv2.imshow('curvature_map', np.abs(curvature_img) * 100.0)
    cv2.imshow('debug', color_img)
    # cv2.imshow('threshold', thresh_img)
    cv2.imshow('curvature_mag', curvature_mag_img)
    cv2.imshow('dist', dist_transform * 1000)

    #cv2.imshow('sure_fg', sure_fg)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    #break

    # time.sleep(100.0)
# cam.release()
cv2.destroyAllWindows()