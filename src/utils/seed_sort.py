# -*- coding: utf-8 -*-
# @author: Weimin Zhang (weiminzhang199205@163.com)
# @date: 19/4/2 22:02
# @version: 1.0

import numpy as np
import copy


def partition(array, l, r, sort_key_idx):
    # l, r = 0, len(array) - 1
    if l > r:
        return l
    # piv = array[l]
    piv = copy.deepcopy(array[l])

    while l < r:
        while l < r and array[r][sort_key_idx] <= piv[sort_key_idx]:
            r -= 1
        # array[l] = array[r]
        array[l] = copy.deepcopy(array[r])

        while l < r and array[l][sort_key_idx] >= piv[sort_key_idx]:
            l += 1
        # array[r] =array[l]
        array[r] = copy.deepcopy(array[l])
    array[l] = copy.deepcopy(piv)
    # array[l] = piv


    return l


def qsort(array, sort_key_idx, l, r):
    if l >= r:
        return array
    piv = partition(array, l, r, sort_key_idx)
    if l < piv:
        array = qsort(array, sort_key_idx, l, piv - 1)
    if piv < r:
        array = qsort(array, sort_key_idx, piv + 1, r)
    return array


def qsort_seed(array, sort_key_idx, seed=0):
    import numpy as npx

    random = npx.random.RandomState(seed)

    random.shuffle(array)
    # my qsort
    return qsort(array, sort_key_idx, 0, len(array) - 1)


def sort_action_space_by_pr(action_space, pr_scores, seed):
    array = [(i, pr_scores[_[1]]) for i, _ in enumerate(action_space)]
    array = np.asarray(array )
    array = qsort_seed(array, 1, seed)
    idx = array[:,0].astype(np.int32)
    print idx
    return action_space[idx]

def run():
    vs = [7, 1, 3, 4, 2, 9, 10, 8]

    array = [[i, _] for i, _ in enumerate(vs)]
    array2 = [[i, 1] for i, _ in enumerate(vs)]
    array, array2 = np.array(array), np.array(array2)
    print qsort_seed(array, 1, seed=10)
    print "seed 10:", qsort_seed(copy.deepcopy(array2), 1, seed=10)
    print "seed 10:", qsort_seed(copy.deepcopy(array2), 1, seed=10)
    print "seed 11:", qsort_seed(copy.deepcopy(array2), 1, seed=11)
    print "seed 11:", qsort_seed(copy.deepcopy(array2), 1, seed=11)

def test2():
    action_space = [ (1, 3), (3,2), (4,5) ]
    action_space = np.array(action_space, dtype=np.int32)
    prs = {3:0.5,2:0.1,5:0.6}
    action_space = sort_action_space_by_pr(action_space, prs, 1)
    print action_space

if __name__ == '__main__':
    run()
    # test2()

__END__ = True
