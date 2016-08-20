import cv2
import numpy as np

camera_port = -1
cam = None

if camera_port >= 0:
    cam = cv2.VideoCapture(camera_port)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cam.set(cv2.CAP_PROP_FPS, 30)
else:
    frame = cv2.imread('testdata/stones.jpg')

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
    curvature_img = np.zeros_like(gray_img, dtype=np.float)

    concave_points = []
    curvature_step = 7

    for contour in contours:
        c_len = len(contour)
        n = curvature_step

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

            if (angle > 0):
                normal = v_a / mag_a + v_b / mag_b
                normal = normal / np.linalg.norm(normal)
                concave_points.append([p_b, normal, angle])

            curvature_img[p_b[1], p_b[0]] = angle * 0.5   # TODO: can go

    # Draw concave normals
    cv2.drawContours(color_img, contours, -1, (255, 0, 0))

    normal_draw_length = 10.0
    for pt, normal, angle in concave_points:
        n = np.uint16(np.round(pt + normal * normal_draw_length * angle))

        cv2.line(color_img, (pt[0], pt[1]), (n[0], n[1]), (0, 0, 255))

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