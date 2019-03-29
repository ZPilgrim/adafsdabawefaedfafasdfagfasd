# -*- coding: utf-8 -*-
# @author: Weimin Zhang (weiminzhang199205@163.com)
# @date: 19/3/21 上午12:11
# @version: 1.0

import pickle as pkl
from collections import defaultdict


def load(fn):
    return pkl.load(open(fn, 'rb'))


def build(adj, e2t):
    ret = defaultdict()
    for src in adj:
        src_abs = e2t[src]
        if src_abs not in ret:
            ret[src_abs] = defaultdict()
        for r in adj[src]:
            if r not in ret[src_abs]:
                ret[src_abs][r] = set()
            for v in adj[src][r]:
                ret[src_abs][r].add(e2t[v])

    return ret


def run():
    data_dir = '../../data/NELL-995/'
    adj = load(data_dir + 'adj_list.pkl')
    e2t = load(data_dir + 'entity2typeid.pkl')
    ret = build(adj, e2t)
    open(data_dir + 'adj_list_abs.pkl', 'wb').write(pkl.dumps(ret))


if __name__ == '__main__':
    run()

__END__ = True
