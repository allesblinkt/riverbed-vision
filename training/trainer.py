import os
import pickle as serialization
import sys

sys.path.insert(0, '../brain')

from coloranalysis import compare_colors
from structure import compare_histograms


from log import makelog
log = makelog(__name__)

STONE_DATA_PATH = '../brain/stones/'



def load_stones():
	stones = []

	for fn in os.listdir(STONE_DATA_PATH):
		if fn.endswith('.data'):
			path = os.path.join(STONE_DATA_PATH, fn)
			with open(path, 'rb') as f:
				ident = os.path.splitext(fn)[0]

				stone = serialization.load(f)
				stone.identifier = ident
				stones.append(stone)

	log.info('Loaded %d stones', len(stones))

	return stones



def find_best_matches(stone, selection, target_count=10):    # TODO: put selection in one place
    bucket_size_c=10 * target_count
    bucket_size_s=200 * target_count

    min_colors = sorted(selection, key=lambda x: compare_colors(x.color, stone.color))
    min_structures = sorted(selection, key=lambda x: compare_histograms(x.structure, stone.structure))

    color_set = set(min_colors[:bucket_size_c])
    structure_set = set(min_structures[:bucket_size_s])

    intersection_list = list(color_set.intersection(structure_set))

    if len(intersection_list) > 0:
        log.debug('Found stone matches in intersection set')
        return intersection_list

    return min_colors

