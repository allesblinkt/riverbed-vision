"""
This implementation of SFTA is an adaption of sfta.py from
https://github.com/ebenolson/tessa

Original idea and implementation
Alceu Costa
"An Efficient Algorithm for Fractal Analysis of Textures"
"""

import numpy as np
import cv2


def find_borders(img):
    """Find borders in a threshold image and highlight them."""
    pad_img = np.pad(img, [[1, 1], [1, 1]], 'constant', constant_values=1).astype('uint8')

    borders = pad_img[2:, 1:-1] + pad_img[0:-2, 1:-1] +  \
        pad_img[1:-1:, 2:] + pad_img[1:-1:, 0:-2] + \
        pad_img[2:, 2:] + pad_img[2:, 0:-2] + \
        pad_img[0:-2, 2:] + pad_img[0:-2, 0:-2]
    return img * (borders < 8)


def otsu(counts):
    p = counts * 1.0 / np.sum(counts)
    omega = np.cumsum(p)
    mu = np.cumsum(p * range(1, len(p) + 1))
    mu_t = mu[-1]

    sigma_b_squared = (mu_t * omega - mu)**2 / (omega * (1 - omega))
    maxval = np.max(np.nan_to_num(sigma_b_squared))
    if np.isnan(sigma_b_squared).all():
        pos = 0
    else:
        pos = np.mean((sigma_b_squared == maxval).nonzero()) + 1
    return pos


def otsu_recurse(img, ttotal):
    if img == []:
        return []

    img = img.astype(np.uint8).flatten()

    num_bins = 256
    counts = np.histogram(img, range(num_bins))[0]

    thresholds = np.zeros((ttotal, 1))

    def otsu_recurse_step(lower_bin, upper_bin, lower_thresh, upper_thresh):
        if ((upper_thresh < lower_thresh) or (lower_bin >= upper_bin)):
            return

        level = otsu(counts[int(np.ceil(lower_bin)) - 1:int(np.ceil(upper_bin))]) + lower_bin

        pos_insert = int(np.ceil((lower_thresh + upper_thresh) / 2.0))
        thresholds[pos_insert - 1] = level / num_bins
        otsu_recurse_step(lower_bin, level, lower_thresh, pos_insert - 1)
        otsu_recurse_step(level + 1, upper_bin, pos_insert + 1, upper_thresh)

    otsu_recurse_step(1, num_bins, 1, ttotal)
    return [t[0] for t in thresholds]


def haus_dim(img):
    """Returns the Haussdorf fractal dimension of an object represented by a binary image."""
    dim_max = np.max(np.shape(img))
    dim_new = int(2**np.ceil(np.log2(dim_max)))
    rows_padded = dim_new - np.shape(img)[0]
    cols_padded = dim_new - np.shape(img)[1]

    img = np.pad(img, ((0, rows_padded), (0, cols_padded)), 'constant')

    box_counts = np.zeros(int(np.ceil(np.log2(dim_max))) + 1)
    resolutions = np.zeros(int(np.ceil(np.log2(dim_max))) + 1)

    img_size = np.shape(img)[0]
    box_size = 1
    idx = 0
    while box_size <= img_size:
        box_count = (img > 0).sum()
        idx = idx + 1
        box_counts[idx - 1] = box_count
        resolutions[idx - 1] = 1.0 / box_size

        box_size = box_size * 2
        img = img[::2, ::2] + img[1::2, ::2] + img[1::2, 1::2] + img[::2, 1::2]

    return np.polyfit(np.log(resolutions), np.log(box_counts), 1)[0]


def sfta(img, nt):
    if len(np.shape(img)) == 3:
        img = np.mean(img, 2)
    elif len(np.shape(img)) != 2:
        raise ImageDimensionError

    img = img.astype(np.uint8)

    thresholds = otsu_recurse(img, nt)
    print(thresholds)

    size_feat_vect = len(thresholds) * 6
    feat_vect = np.zeros(size_feat_vect)

    debug_imgs = []

    vect_idx = 0
    for t in range(len(thresholds)):
        thresh = thresholds[t]
        img_borders = img > (thresh * 255)
        debug_imgs.append(img_borders.copy())

        img_borders = find_borders(img_borders)

        vals = img[img_borders.nonzero()].astype(np.double)
        feat_vect[vect_idx + 0] = haus_dim(img_borders)
        feat_vect[vect_idx + 1] = np.mean(vals)
        feat_vect[vect_idx + 2] = len(vals)
        vect_idx += 3

    thresholds = thresholds + [1.0, ]
    for t in range(len(thresholds) - 1):
        lower_thresh = thresholds[t]
        upper_thresh = thresholds[t + 1]
        img_borders = (img > (lower_thresh * 255)) * (img < (upper_thresh * 255))
        debug_imgs.append(img_borders.copy())

        img_borders = find_borders(img_borders)

        vals = img[img_borders.nonzero()].astype(np.double)
        feat_vect[vect_idx + 0] = haus_dim(img_borders)
        feat_vect[vect_idx + 1] = np.mean(vals)
        feat_vect[vect_idx + 2] = len(vals)
        vect_idx += 3
    return feat_vect, debug_imgs


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    pics = ['stone1.png', 'stone2.png', 'stone3.png', 'stone4.png', 'stone5.png']

    for p in pics:
        img = cv2.imread(p)
        D, imgs = sfta(img, 10)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

        #plt.imshow(hist, interpolation='nearest')
        cv2.imshow("hist  " + p, hist/255.0)
        #plt.show()

        # for i in range(len(imgs)):
        #     print(imgs[i])
        #     nimg = np.array(imgs[i], dtype=np.uint8) * 255
        #     print(nimg)
        #     cv2.imshow("a", nimg)
        #     cv2.waitKey(0)
    cv2.waitKey(0)
