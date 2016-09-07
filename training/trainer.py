import os
import pickle as serialization
import sys

import cv2
import numpy as np

sys.path.insert(0, '../brain')

from coloranalysis import compare_colors, histogram_lab
from structure import compare_histograms, lbp_histogram
from structure_sfta import sfta

from featureanalysis import image_patches

from log import makelog
log = makelog(__name__)

STONE_DATA_PATH = '../brain/stones/'

from sklearn.utils import assert_all_finite

def load_stones():
    stones = {}

    for fn in os.listdir(STONE_DATA_PATH):
        if fn.endswith('.data'):
            path = os.path.join(STONE_DATA_PATH, fn)
            with open(path, 'rb') as f:
                ident = os.path.splitext(fn)[0]

                stone = serialization.load(f)
                stone.identifier = ident
                # stones.append(stone)

                stones[stone.identifier] = stone

    log.info('Loaded %d stones', len(stones))

    return stones


def find_best_matches(stone, selection, target_count=10):    # TODO: put selection in one place
    bucket_size_c = 20 * target_count
    bucket_size_s = 100 * target_count

    min_colors = sorted(selection, key=lambda x: compare_colors(x.color, stone.color))
    min_structures = sorted(selection, key=lambda x: compare_histograms(x.structure, stone.structure))

    color_set = set(min_colors[:bucket_size_c])
    structure_set = set(min_structures[:bucket_size_s])

    intersection_list = list(color_set.intersection(structure_set))

    union_list = []
    union_list.extend(min_colors[:target_count // 2])
    #union_list.extend(min_structures[:target_count // 2])

    if False and len(intersection_list) > 0:
        log.debug('Found stone matches in intersection set')
        intersection_list = sorted(intersection_list, key=lambda x: compare_colors(x.color, stone.color))

        # intersection_list = sorted(intersection_list, key=lambda x: compare_histograms(x.structure, stone.structure))
        return intersection_list

    return union_list


def max_occurrences(seq):
    from operator import itemgetter

    c = dict()
    for item in seq:
        c[item] = c.get(item, 0) + 1
    return max(c.items(), key=itemgetter(1))



def stone_do(stone, draw=False):


    fn = stone.identifier + '.png'
    path = os.path.join('static/stones/', fn)

    print("Processing %s" % (stone.identifier, ))

    stone_img = cv2.imread(path, -1)
    patches = image_patches(stone_img, patch_size=(64, 64), patch_step=5, max_count=10)
    print(len(patches))

    feat_vecs = []
    histos = []

    if len(patches) == 0:
        return []

    for patch in patches:
        D, simgs = sfta(patch, 8)
        hist = histogram_lab(patch)
        lbp_hist = lbp_histogram(patch)

        if draw:
            cv2.imshow("p", patch)

            for i in range(len(simgs)):
                simg = simgs[i]
                nimg = np.array(simg, dtype=np.uint8) * 255
                cv2.imshow("simg" + str(i), nimg)

            cv2.waitKey(1)

        print(lbp_hist.shape)
        # combined = np.concatenate((D, hist, lbp_hist), axis=0)
        combined = np.concatenate((D, hist), axis=0)

        # print("Combinded: ", combined)
        feat_vecs.append(combined)

    feat_vecs = np.array(feat_vecs)
    feat_vec = feat_vecs.sum(axis=0)
    
    # if (feat_vec.shape[0] != 36):
    #     gargler()

    if (feat_vecs.shape[0] > 0):
        feat_vec = feat_vec / float(feat_vecs.shape[0])

    try:
        assert_all_finite(feat_vecs)
        assert_all_finite(feat_vec)
    except:
        print(feat_vecs)
        print(feat_vec)
        cv2.waitKey(0)

    return [feat_vec]


def extract_features(stones, stone_category_votes):
    training = []
    cnt = 0
    for stone_id in stone_category_votes:
        votes = stone_category_votes[stone_id]

        if len(votes) == 0:
            continue

        label, label_count = max_occurrences(votes)
        print("# # " + label)
        feat_vecs = stone_do(stones[stone_id], draw=True)

        print(" ")

        for v in feat_vecs:
            training.append((v, label))

        # cnt = cnt + 1

        # if cnt > 4:
        #     break

    return training


if __name__ == '__main__':
    import shelve

    stones = load_stones()
    db_features = shelve.open('features.db', 'c')

    if False:
        db_voting = shelve.open('voting.db', 'r')

        stone_category_votes = db_voting["stone_category_votes"]
        training = extract_features(stones, stone_category_votes)

        db_features["training"] = training
        db_voting.close()

    if True:
        training = db_features["training"]
        valid_training = []

        for t in training:
            print("")

            try:
                assert_all_finite(np.asarray(t))
                valid_training.append(t)
            except:
                print("Fnackadu")
                print(t)

            # print(t)
            # #print(data[i])
            # print(labels[i])
            # print(len(data[i]))
            # print(max(data[i]))

        data = [e[0] for e in valid_training]
        labels = [e[1] for e in valid_training]
        print(np.array(data))

        data = np.asarray(data)
        labels = np.array(labels)
        print(data.shape)

        # data.shape() = np.array(data)
        # labels
        # # for i in range(labels.shape[0]):
        #     print(labels[i])

        # #     print("====")
        # #     print(d)

        # print(data)

        # print("Training with %d patches" % data.shape[0])
        # print("Training with %d patches" % data.shape[1])

        assert_all_finite(data)
        from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import SVC

        gnb = GaussianNB()
        gnb.fit(data, labels)
        y_pred = gnb.predict(data)

        clf = SVC(decision_function_shape='ovo')
        clf.fit(data, labels)
        y2_pred = clf.predict(data)


        print("Number of mislabeled points out of a total %d points : %d" % (data.shape[0], (labels != y_pred).sum()))
        print("Number of mislabeled points out of a total %d points : %d" % (data.shape[0], (labels != y2_pred).sum()))
        
        import time
        time.sleep(5.0)


        for stone_id in stones:
            stone = stones[stone_id]
            feat_l = stone_do(stone, draw=False)
            # print(feat_l)

            fn = stone.identifier + '.png'
            path = os.path.join('static/stones/', fn)
            stone_img = cv2.imread(path, -1)
            pred = gnb.predict(np.array(feat_l))
            print(pred)
            pred2 = clf.predict(np.array(feat_l))
            print(pred2)
            cv2.imshow("stone", stone_img)
            cv2.waitKey(0)

        
    db_voting.close()
    db_features.close()
