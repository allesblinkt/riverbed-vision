import cv2
import numpy as np


def kmeans_quantization(img, n_clusters):
    """ Quantizises the image into a given number of clusters (n_clusters). 
        This can work with images (4 channel) that have an alpha channel, this gets ignored,
        but it will "spend" one cluster for that 
    """
    has_mask = img.shape[2] == 4

    color = img[:, :, 0:3] if has_mask else img
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)    # Convert to lab for better perceptual accuracy
    lab = lab.reshape((color.shape[0] * color.shape[1], 3))

    flab = np.float32(lab)

    if has_mask:
        mask = img[:, :, 3]
        mask = mask.reshape((mask.shape[0] * mask.shape[1], 1))
        flab[np.where(mask == 0)[0]] = (-255, -255, -255)  # Mask off alpha areas

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv2.kmeans(flab, n_clusters, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    exclude_idx = np.where(centers[:, -1] < 0)[0]
    center = np.uint8(centers)

    res = center[labels.flatten()]
    res2 = res.reshape((color.shape))
    res2 = cv2.cvtColor(res2, cv2.COLOR_LAB2BGR)

    hist = centroid_histogram(labels)
    hist[exclude_idx] = 0.0

    dominant = center[np.argmax(hist)]

    dummy = np.array([np.array([dominant])])
    dominant = cv2.cvtColor(dummy, cv2.COLOR_LAB2BGR)[0, 0]

    return dominant, hist, res2


def centroid_histogram(labels):
    num_labels = np.arange(0, len(np.unique(labels)) + 1)
    (hist, _) = np.histogram(labels, bins=num_labels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def find_dominant_color(img):
    """ Expects an image with or without alpha channel and returns the dominant color (BGR) """

    small_img = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

    dominant, hist, processed = kmeans_quantization(small_img, n_clusters=3)
    return dominant.tolist()


if __name__ == '__main__':
    import os
    import fnmatch
    import time

    p = "."   # looks here for pngs...

    pngfiles = []

    for file in os.listdir(p):
        if fnmatch.fnmatch(file, '*.png'):
            pngfiles.append(file)

    for fn in pngfiles:
        print fn
        image = cv2.imread(os.path.join(p, fn), -1)
        (h, w) = image.shape[:2]

        t = time.time()
        dominant = find_dominant_color(image)
        cv2.circle(image, (w / 2, h / 2), w / 8, dominant, -1)
        print("Time taken: %.3  `f" % (time.time() - t))

        cv2.imshow("image", image)
        cv2.waitKey(0)
