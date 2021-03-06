# -*- coding: utf-8 -*-
"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Data processing utilities.
"""

import collections
import numpy as np
import os
import pickle
import copy

START_RELATION = 'START_RELATION'
NO_OP_RELATION = 'NO_OP_RELATION'
NO_OP_ENTITY = 'NO_OP_ENTITY'
DUMMY_RELATION = 'DUMMY_RELATION'
DUMMY_ENTITY = 'DUMMY_ENTITY'

DUMMY_RELATION_ID = 0
START_RELATION_ID = 1
NO_OP_RELATION_ID = 2
DUMMY_ENTITY_ID = 0
NO_OP_ENTITY_ID = 1


def build(adj, e2t):
    ret = collections.defaultdict()
    for src in adj:
        src_abs = e2t[src]
        if src_abs not in ret:
            ret[src_abs] = collections.defaultdict()
        for r in adj[src]:
            if r not in ret[src_abs]:
                ret[src_abs][r] = set()
            for v in adj[src][r]:
                ret[src_abs][r].add(e2t[v])

    return ret


def build_e2t(adj, e2t):
    ret = collections.defaultdict()
    for src in adj:
        #src_abs = e2t[src]
        if src not in ret:
            ret[src] = collections.defaultdict()
        for r in adj[src]:
            if r not in ret[src]:
                ret[src][r] = []
            for v in adj[src][r]:
                ret[src][r].append(e2t[v])
    return ret

# def build_e2t(adj, e2t, bandwidth, data_dir):
#     def load_page_rank_scores(input_path, entity2id):
#         pgrk_scores = collections.defaultdict(float)
#         with open(input_path) as f:
#             for line in f:
#                 e, score = line.strip().split(':')
#                 e_id = entity2id[e.strip()]
#                 score = float(score)
#                 pgrk_scores[e_id] = score
#         return pgrk_scores  
    
#     entity2id, _ = load_index(os.path.join(data_dir, 'entity2id.txt'))
#     page_rank_scores = load_page_rank_scores(
#         os.path.join(data_dir, 'raw.pgrk'), entity2id)

#     ret = collections.defaultdict()
#     if bandwidth:
#         print("building e2t abs graph for training with bandwith {}".format(bandwidth))
#     else:
#         print("building e2t abs graph for training without bandwith")
#     print(bandwidth)
#     for src in adj:
#         #src_abs = e2t[src]
#         if src not in ret:
#             ret[src] = collections.defaultdict()
#         temp = []
#         for r in adj[src]:
#             targets = adj[src][r]
#             for e2 in targets:
#                 temp.append((r, e2))
#         #截取
#         if bandwidth and len(temp) + 1 >= bandwidth:
#             _ = sorted(temp, key=lambda x: page_rank_scores[x[1]], reverse=True)
#             temp = _[ :bandwidth]
#         # else: 
#         #     temp = sorted(
#         #         temp, key=lambda x: page_rank_scores[x[1]], reverse=True)
#         #用截取后的tuple填充adj
        
   
#         assert(len(temp) <=  bandwidth )
#         for _ in temp:
#             r, e = _
#             if r not in ret[src]:
#                 ret[src][r] = set()
#             ret[src][r].add(e2t[e])
#                 #assert(len(ret[src][r]) <= bandwidth)
#     num_facts_e2t = 0
#     out_degrees_e2t = collections.defaultdict(int)
#     for e1 in ret:
#         for r in ret[e1]:
#             num_facts_e2t += len(ret[e1][r])
#             out_degrees_e2t[e1] += len(ret[e1][r])
    
#     print("Sanity check: maximum out degree: {}".format(max(out_degrees_e2t.values())))

#     return ret


def check_answer_ratio(examples):
    entity_dict = {}
    for e1, e2, r in examples:
        if not e1 in entity_dict:
            entity_dict[e1] = set()
        entity_dict[e1].add(e2)
    answer_ratio = 0
    for e1 in entity_dict:
        answer_ratio += len(entity_dict[e1])
    return answer_ratio / len(entity_dict)


def check_relation_answer_ratio(input_file, kg):
    example_dict = {}
    with open(input_file) as f:
        for line in f:
            e1, e2, r = line.strip().split()
            e1 = kg.entity2id[e1]
            e2 = kg.entity2id[e2]
            r = kg.relation2id[r]
            if not r in example_dict:
                example_dict[r] = []
            example_dict[r].append((e1, e2, r))
    r_answer_ratio = {}
    for r in example_dict:
        r_answer_ratio[r] = check_answer_ratio(example_dict[r])
    return r_answer_ratio


def change_to_test_model_path(dataset, model_path):
    model_dir = os.path.dirname(os.path.dirname(model_path))
    model_subdir = os.path.basename(os.path.dirname(model_path))
    file_name = os.path.basename(model_path)
    new_model_subdir = dataset + '.test' + model_subdir[len(dataset):]
    new_model_subdir += '-test'
    new_model_path = os.path.join(model_dir, new_model_subdir, file_name)
    return new_model_path


def get_train_path(args):
    if 'NELL' in args.data_dir:
        if not args.model.startswith('point'):
            if args.test:
                train_path = os.path.join(args.data_dir, 'train.dev.large.triples')
            else:
                train_path = os.path.join(args.data_dir, 'train.large.triples')
        else:
            if args.test:
                print("NELL point else get test path")
                train_path = os.path.join(args.data_dir, 'train.dev.triples')
            else:
                print("NELL point else get train path")
                train_path = os.path.join(args.data_dir, 'train.triples')
    else:
        train_path = os.path.join(args.data_dir, 'train.triples')

    return train_path


def load_seen_entities(adj_list_path, entity_index_path):
    _, id2entity = load_index(entity_index_path)
    with open(adj_list_path, 'rb') as f:
        adj_list = pickle.load(f)
    seen_entities = set()
    for e1 in adj_list:
        seen_entities.add(id2entity[e1])
        for r in adj_list[e1]:
            for e2 in adj_list[e1][r]:
                seen_entities.add(id2entity[e2])
    print('{} seen entities loaded...'.format(len(seen_entities)))
    return seen_entities


# def load_seen_entities_with_abs(adj_list_path, entity_index_path, entity2typeid):
#     _, id2entity = load_index(entity_index_path)
#     with open(adj_list_path, 'rb') as f:
#         adj_list = pickle.load(f)
#     seen_entities = set()
#     seen_entities_type = set()
#     for e1 in adj_list:
#         seen_entities.add(id2entity[e1])
#         seen_entities_type.add(entity2typeid(id2entity[e1]))
#         for r in adj_list[e1]:
#             for e2 in adj_list[e1][r]:
#                 seen_entities.add(id2entity[e2])
#     print('{} seen entities loaded...'.format(len(seen_entities)))
#     return seen_entities


def load_triples_with_label(data_path, r, entity_index_path, relation_index_path, seen_entities=None, verbose=False):
    entity2id, _ = load_index(entity_index_path)
    relation2id, _ = load_index(relation_index_path)

    def triple2ids(e1, e2, r):
        return entity2id[e1], entity2id[e2], relation2id[r]

    triples, labels = [], []
    with open(data_path) as f:
        num_skipped = 0
        for line in f:
            pair, label = line.strip().split(': ')
            e1, e2 = pair.strip().split(',')
            if seen_entities and (not e1 in seen_entities or not e2 in seen_entities):
                num_skipped += 1
                if verbose:
                    print('Skip triple ({}) with unseen entity: {}'.format(num_skipped, line.strip()))
                continue
            triples.append(triple2ids(e1, e2, r))
            labels.append(label.strip())
    return triples, labels


def load_triples(data_path, entity_index_path, relation_index_path, group_examples_by_query=False,
                 add_reverse_relations=False, seen_entities=None, verbose=False):
    """
    Convert triples stored on disc into indices.
    """
    entity2id, _ = load_index(entity_index_path)
    relation2id, _ = load_index(relation_index_path)

    def triple2ids(e1, e2, r):
        return entity2id[e1], entity2id[e2], relation2id[r]

    triples = []
    if group_examples_by_query:
        triple_dict = {}
    with open(data_path) as f:
        num_skipped = 0
        for line in f:
            e1, e2, r = line.strip().split()
            if seen_entities and (not e1 in seen_entities or not e2 in seen_entities):
                num_skipped += 1
                if verbose:
                    print('Skip triple ({}) with unseen entity: {}'.format(num_skipped, line.strip()))
                continue
            # if r in ['concept:agentbelongstoorganization', 'concept:teamplaysinleague']:
            #     continue
            if group_examples_by_query:
                e1_id, e2_id, r_id = triple2ids(e1, e2, r)
                if e1_id not in triple_dict:
                    triple_dict[e1_id] = {}
                if r_id not in triple_dict[e1_id]:
                    triple_dict[e1_id][r_id] = set()
                triple_dict[e1_id][r_id].add(e2_id)
                if add_reverse_relations:
                    r_inv = r + '_inv'
                    e2_id, e1_id, r_inv_id = triple2ids(e2, e1, r_inv)
                    if e2_id not in triple_dict:
                        triple_dict[e2_id] = {}
                    if r_inv_id not in triple_dict[e2_id]:
                        triple_dict[e2_id][r_inv_id] = set()
                    triple_dict[e2_id][r_inv_id].add(e1_id)
            else:
                triples.append(triple2ids(e1, e2, r))
                if add_reverse_relations:
                    triples.append(triple2ids(e2, e1, r + '_inv'))
    if group_examples_by_query:
        for e1_id in triple_dict:
            for r_id in triple_dict[e1_id]:
                triples.append((e1_id, list(triple_dict[e1_id][r_id]), r_id))
    print('{} triples loaded from {}'.format(len(triples), data_path))
    return triples


def convert_entities2typeids(triples, entity2typeid):
    ret = []
    for e1, e2, rs in triples:
        if type(e2) == list:
            es = [entity2typeid[_] for _ in e2]
        else:
            es = entity2typeid[e2]
        # if entity2typeid[e1] > 270 or entity2typeid[e2] > 270:
        #     print("ERROR etype:", e1, e2, entity2typeid[e1], entity2typeid[e2])
        ret.append(
            (entity2typeid[e1], es, rs)
        )

    return ret


def load_entity_hist(input_path):
    entity_hist = {}
    with open(input_path) as f:
        for line in f.readlines():
            v, f = line.strip().split()
            entity_hist[v] = int(f)
    return entity_hist


def load_index(input_path):
    index, rev_index = {}, {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):
            v, _ = line.strip().split()
            index[v] = i
            rev_index[i] = v
    return index, rev_index


def prepare_kb_envrioment(raw_kb_path, train_path, dev_path, test_path, test_mode, bandwidth, add_reverse_relations=True):
    """
    Process KB data which was saved as a set of triples.
        (a) Remove train and test triples from the KB envrionment.
        (b) Add reverse triples on demand.
        (c) Index unique entities and relations appeared in the KB.

    :param raw_kb_path: Path to the raw KB triples.
    :param train_path: Path to the train set KB triples.
    :param dev_path: Path to the dev set KB triples.
    :param test_path: Path to the test set KB triples.
    :param add_reverse_relations: If set, add reverse triples to the KB environment.
    """
    data_dir = os.path.dirname(raw_kb_path)

    def get_type(e_name):
        if e_name == DUMMY_ENTITY:
            return DUMMY_ENTITY_ID
        if e_name == NO_OP_ENTITY:
            return NO_OP_ENTITY_ID
        if 'nell-995' in data_dir.lower():
            if '_' in e_name:
                return e_name.split('_')[1]
            else:
                return 'numerical'
        else:
            assert (1 == 0)  # 针对其他数据集这里需要重新get type
            return 'entity'

    def hist_to_vocab(_dict):
        return sorted(sorted(_dict.items(), key=lambda x: x[0]), key=lambda x: x[1], reverse=True)

    # Create entity and relation indices
    entity_hist = collections.defaultdict(int)
    relation_hist = collections.defaultdict(int)
    type_hist = collections.defaultdict(int)
    with open(raw_kb_path) as f:
        raw_kb_triples = [l.strip() for l in f.readlines()]
    with open(train_path) as f:
        train_triples = [l.strip() for l in f.readlines()]
    with open(dev_path) as f:
        dev_triples = [l.strip() for l in f.readlines()]
    with open(test_path) as f:
        test_triples = [l.strip() for l in f.readlines()]

    if test_mode:
        keep_triples = train_triples + dev_triples
        removed_triples = test_triples
    else:
        keep_triples = train_triples
        removed_triples = dev_triples + test_triples

    # Index entities and relations
    for line in set(raw_kb_triples + keep_triples + removed_triples):
        e1, e2, r = line.strip().split()
        entity_hist[e1] += 1
        entity_hist[e2] += 1
        if 'nell-995' in data_dir.lower():
            # 在训练样本中不会出现dummy和no_op，所以不影响
            t1 = e1.split('_')[1] if '_' in e1 else 'numerical'
            t2 = e2.split('_')[1] if '_' in e2 else 'numerical'
        else:
            t1 = get_type(e1)
            t2 = get_type(e2)
        type_hist[t1] += 1
        type_hist[t2] += 1
        relation_hist[r] += 1
        if add_reverse_relations:
            inv_r = r + '_inv'
            relation_hist[inv_r] += 1
    # Save the entity and relation indices sorted by decreasing frequency
    with open(os.path.join(data_dir, 'entity2id.txt'), 'w') as o_f:
        o_f.write('{}\t{}\n'.format(DUMMY_ENTITY, DUMMY_ENTITY_ID))
        o_f.write('{}\t{}\n'.format(NO_OP_ENTITY, NO_OP_ENTITY_ID))
        for e, freq in hist_to_vocab(entity_hist):
            o_f.write('{}\t{}\n'.format(e, freq))
    with open(os.path.join(data_dir, 'relation2id.txt'), 'w') as o_f:
        o_f.write('{}\t{}\n'.format(DUMMY_RELATION, DUMMY_RELATION_ID))
        o_f.write('{}\t{}\n'.format(START_RELATION, START_RELATION_ID))
        o_f.write('{}\t{}\n'.format(NO_OP_RELATION, NO_OP_RELATION_ID))
        for r, freq in hist_to_vocab(relation_hist):
            o_f.write('{}\t{}\n'.format(r, freq))
    with open(os.path.join(data_dir, 'type2id.txt'), 'w') as o_f:
        o_f.write('{}\t{}\n'.format(DUMMY_ENTITY, DUMMY_ENTITY_ID))
        o_f.write('{}\t{}\n'.format(NO_OP_ENTITY, NO_OP_ENTITY_ID))
        for t, freq in hist_to_vocab(type_hist):
            o_f.write('{}\t{}\n'.format(t, freq))
    print('{} entities indexed'.format(len(entity_hist)))
    print('{} relations indexed'.format(len(relation_hist)))
    print('{} types indexed'.format(len(type_hist)))
    entity2id, id2entity = load_index(os.path.join(data_dir, 'entity2id.txt'))
    relation2id, id2relation = load_index(os.path.join(data_dir, 'relation2id.txt'))
    type2id, id2type = load_index(os.path.join(data_dir, 'type2id.txt'))

    removed_triples = set(removed_triples)
    adj_list = collections.defaultdict(collections.defaultdict)
    entity2typeid = [0 for i in range(len(entity2id))]
    entity2typeid[DUMMY_ENTITY_ID] = DUMMY_ENTITY_ID
    entity2typeid[NO_OP_ENTITY_ID] = NO_OP_ENTITY_ID
    num_facts = 0
    for line in set(raw_kb_triples + keep_triples):
        e1, e2, r = line.strip().split()
        triple_signature = '{}\t{}\t{}'.format(e1, e2, r)
        e1_id = entity2id[e1]
        e2_id = entity2id[e2]
        t1 = get_type(e1)
        t2 = get_type(e2)
        t1_id = type2id[t1]
        t2_id = type2id[t2]
        entity2typeid[e1_id] = t1_id
        entity2typeid[e2_id] = t2_id
        if not triple_signature in removed_triples:
            r_id = relation2id[r]
            if not r_id in adj_list[e1_id]:
                adj_list[e1_id][r_id] = set()
            if e2_id in adj_list[e1_id][r_id]:
                print('Duplicate fact: {} ({}, {}, {})!'.format(
                    line.strip(), id2entity[e1_id], id2relation[r_id], id2entity[e2_id]))
            adj_list[e1_id][r_id].add(e2_id)
            num_facts += 1
            if add_reverse_relations:
                inv_r = r + '_inv'
                inv_r_id = relation2id[inv_r]
                if not inv_r_id in adj_list[e2_id]:
                    adj_list[e2_id][inv_r_id] = set([])
                if e1_id in adj_list[e2_id][inv_r_id]:
                    print('Duplicate fact: {} ({}, {}, {})!'.format(
                        line.strip(), id2entity[e2_id], id2relation[inv_r_id], id2entity[e1_id]))
                adj_list[e2_id][inv_r_id].add(e1_id)
                num_facts += 1
    print('{} facts processed'.format(num_facts))
    # Save adjacency list

    # Convert dict of dict with set value to dict of dict with list value
    for src in adj_list:
        for r in adj_list[src]:
            adj_list[src][r] = [v for v in adj_list[src][r]]

    adj_list_path = os.path.join(data_dir, 'adj_list.pkl')
    with open(adj_list_path, 'wb') as o_f:
        pickle.dump(dict(adj_list), o_f)
    with open(os.path.join(data_dir, 'entity2typeid.pkl'), 'wb') as o_f:
        pickle.dump(entity2typeid, o_f)

    print("building abs graph map for test ...........")
    adj_list_abs = build(adj_list, entity2typeid)
    with open(os.path.join(data_dir, 'adj_list_abs.pkl'), 'wb') as o_f:
        o_f.write(pickle.dumps(dict(adj_list_abs)))

    print("building e2t graph map for training ..........")
    adj_list_e2t = build_e2t(adj_list, entity2typeid)
    with open(os.path.join(data_dir, 'adj_list_e2t.pkl'),'wb') as o_f:
        o_f.write(pickle.dumps(dict(adj_list_e2t)))


def prepare_kb_envrioment_with_abs(raw_kb_path, train_path, dev_path, test_path, test_mode, add_reverse_relations=True):
    """
    Process KB data which was saved as a set of triples.
        (a) Remove train and test triples from the KB envrionment.
        (b) Add reverse triples on demand.
        (c) Index unique entities and relations appeared in the KB.

    :param raw_kb_path: Path to the raw KB triples.
    :param train_path: Path to the train set KB triples.
    :param dev_path: Path to the dev set KB triples.
    :param test_path: Path to the test set KB triples.
    :param add_reverse_relations: If set, add reverse triples to the KB environment.
    """
    data_dir = os.path.dirname(raw_kb_path)

    def get_type(e_name):
        if e_name == DUMMY_ENTITY:
            return DUMMY_ENTITY_ID
        if e_name == NO_OP_ENTITY:
            return NO_OP_ENTITY_ID
        if 'nell-995' in data_dir.lower():
            if '_' in e_name:
                return e_name.split('_')[1]
            else:
                return 'numerical'
        else:
            assert (1 == 0)  # 针对其他数据集这里需要重新get type
            return 'entity'

    def hist_to_vocab(_dict):
        return sorted(sorted(_dict.items(), key=lambda x: x[0]), key=lambda x: x[1], reverse=True)

    # Create entity and relation indices
    entity_hist = collections.defaultdict(int)
    relation_hist = collections.defaultdict(int)
    type_hist = collections.defaultdict(int)
    with open(raw_kb_path) as f:
        raw_kb_triples = [l.strip() for l in f.readlines()]
    with open(train_path) as f:
        train_triples = [l.strip() for l in f.readlines()]
    with open(dev_path) as f:
        dev_triples = [l.strip() for l in f.readlines()]
    with open(test_path) as f:
        test_triples = [l.strip() for l in f.readlines()]

    if test_mode:
        keep_triples = train_triples + dev_triples
        removed_triples = test_triples
    else:
        keep_triples = train_triples
        removed_triples = dev_triples + test_triples

    # Index entities and relations
    for line in set(raw_kb_triples + keep_triples + removed_triples):
        e1, e2, r = line.strip().split()
        entity_hist[e1] += 1
        entity_hist[e2] += 1
        if 'nell-995' in data_dir.lower():
            # 在训练样本中不会出现dummy和no_op，所以不影响
            t1 = e1.split('_')[1] if '_' in e1 else 'numerical'
            t2 = e2.split('_')[1] if '_' in e2 else 'numerical'
        else:
            t1 = get_type(e1)
            t2 = get_type(e2)
        type_hist[t1] += 1
        type_hist[t2] += 1
        relation_hist[r] += 1
        if add_reverse_relations:
            inv_r = r + '_inv'
            relation_hist[inv_r] += 1
    # Save the entity and relation indices sorted by decreasing frequency
    with open(os.path.join(data_dir, 'entity2id.txt'), 'w') as o_f:
        o_f.write('{}\t{}\n'.format(DUMMY_ENTITY, DUMMY_ENTITY_ID))
        o_f.write('{}\t{}\n'.format(NO_OP_ENTITY, NO_OP_ENTITY_ID))
        for e, freq in hist_to_vocab(entity_hist):
            o_f.write('{}\t{}\n'.format(e, freq))
    with open(os.path.join(data_dir, 'relation2id.txt'), 'w') as o_f:
        o_f.write('{}\t{}\n'.format(DUMMY_RELATION, DUMMY_RELATION_ID))
        o_f.write('{}\t{}\n'.format(START_RELATION, START_RELATION_ID))
        o_f.write('{}\t{}\n'.format(NO_OP_RELATION, NO_OP_RELATION_ID))
        for r, freq in hist_to_vocab(relation_hist):
            o_f.write('{}\t{}\n'.format(r, freq))
    with open(os.path.join(data_dir, 'type2id.txt'), 'w') as o_f:
        o_f.write('{}\t{}\n'.format(DUMMY_ENTITY, DUMMY_ENTITY_ID))
        o_f.write('{}\t{}\n'.format(NO_OP_ENTITY, NO_OP_ENTITY_ID))
        for t, freq in hist_to_vocab(type_hist):
            o_f.write('{}\t{}\n'.format(t, freq))
    print('{} entities indexed'.format(len(entity_hist)))
    print('{} relations indexed'.format(len(relation_hist)))
    print('{} types indexed'.format(len(type_hist)))
    entity2id, id2entity = load_index(os.path.join(data_dir, 'entity2id.txt'))
    relation2id, id2relation = load_index(os.path.join(data_dir, 'relation2id.txt'))
    type2id, id2type = load_index(os.path.join(data_dir, 'type2id.txt'))

    removed_triples = set(removed_triples)
    adj_list = collections.defaultdict(collections.defaultdict)
    entity2typeid = [0 for i in range(len(entity2id))]
    entity2typeid[DUMMY_ENTITY_ID] = DUMMY_ENTITY_ID
    entity2typeid[NO_OP_ENTITY_ID] = NO_OP_ENTITY_ID
    num_facts = 0
    for line in set(raw_kb_triples + keep_triples):
        e1, e2, r = line.strip().split()
        triple_signature = '{}\t{}\t{}'.format(e1, e2, r)
        e1_id = entity2id[e1]
        e2_id = entity2id[e2]
        t1 = get_type(e1)
        t2 = get_type(e2)
        t1_id = type2id[t1]
        t2_id = type2id[t2]
        entity2typeid[e1_id] = t1_id
        entity2typeid[e2_id] = t2_id
        if not triple_signature in removed_triples:
            r_id = relation2id[r]
            if not r_id in adj_list[e1_id]:
                adj_list[e1_id][r_id] = set()
            if e2_id in adj_list[e1_id][r_id]:
                print('Duplicate fact: {} ({}, {}, {})!'.format(
                    line.strip(), id2entity[e1_id], id2relation[r_id], id2entity[e2_id]))
            adj_list[e1_id][r_id].add(e2_id)
            num_facts += 1
            if add_reverse_relations:
                inv_r = r + '_inv'
                inv_r_id = relation2id[inv_r]
                if not inv_r_id in adj_list[e2_id]:
                    adj_list[e2_id][inv_r_id] = set([])
                if e1_id in adj_list[e2_id][inv_r_id]:
                    print('Duplicate fact: {} ({}, {}, {})!'.format(
                        line.strip(), id2entity[e2_id], id2relation[inv_r_id], id2entity[e1_id]))
                adj_list[e2_id][inv_r_id].add(e1_id)
                num_facts += 1
    print('{} facts processed'.format(num_facts))
    # Save adjacency list
    adj_list_path = os.path.join(data_dir, 'adj_list.pkl')
    with open(adj_list_path, 'wb') as o_f:
        pickle.dump(dict(adj_list), o_f)
    with open(os.path.join(data_dir, 'entity2typeid.pkl'), 'wb') as o_f:
        pickle.dump(entity2typeid, o_f)


def get_seen_queries(data_dir, entity_index_path, relation_index_path):
    entity2id, _ = load_index(entity_index_path)
    relation2id, _ = load_index(relation_index_path)
    seen_queries = set()
    with open(os.path.join(data_dir, 'train.triples')) as f:
        for line in f:
            e1, e2, r = line.strip().split('\t')
            e1_id = entity2id[e1]
            r_id = relation2id[r]
            seen_queries.add((e1_id, r_id))

    seen_exps = []
    unseen_exps = []
    num_exps = 0
    with open(os.path.join(data_dir, 'dev.triples')) as f:
        for line in f:
            num_exps += 1
            e1, e2, r = line.strip().split('\t')
            e1_id = entity2id[e1]
            r_id = relation2id[r]
            if (e1_id, r_id) in seen_queries:
                seen_exps.append(line)
            else:
                unseen_exps.append(line)
    num_seen_exps = len(seen_exps) + 0.0
    num_unseen_exps = len(unseen_exps) + 0.0
    seen_ratio = num_seen_exps / num_exps
    unseen_ratio = num_unseen_exps / num_exps
    print('Seen examples: {}/{} {}'.format(num_seen_exps, num_exps, seen_ratio))
    print('Unseen examples: {}/{} {}'.format(num_unseen_exps, num_exps, unseen_ratio))

    return seen_queries, (seen_ratio, unseen_ratio)


def get_relations_by_type(data_dir, relation_index_path):
    with open(os.path.join(data_dir, 'raw.kb')) as f:
        triples = list(f.readlines())
    with open(os.path.join(data_dir, 'train.triples')) as f:
        triples += list(f.readlines())
    triples = list(set(triples))

    query_answers = dict()

    theta_1_to_M = 1.5

    for triple_str in triples:
        e1, e2, r = triple_str.strip().split('\t')
        if not r in query_answers:
            query_answers[r] = dict()
        if not e1 in query_answers[r]:
            query_answers[r][e1] = set()
        query_answers[r][e1].add(e2)

    to_M_rels = set()
    to_1_rels = set()

    dev_rels = set()
    with open(os.path.join(data_dir, 'dev.triples')) as f:
        for line in f:
            e1, e2, r = line.strip().split('\t')
            dev_rels.add(r)

    relation2id, _ = load_index(relation_index_path)
    num_rels = len(dev_rels)
    print('{} relations in dev dataset in total'.format(num_rels))
    for r in dev_rels:
        ratio = np.mean([len(x) for x in query_answers[r].values()])
        if ratio > theta_1_to_M:
            to_M_rels.add(relation2id[r])
        else:
            to_1_rels.add(relation2id[r])
    num_to_M = len(to_M_rels) + 0.0
    num_to_1 = len(to_1_rels) + 0.0

    print('to-M relations: {}/{} ({})'.format(num_to_M, num_rels, num_to_M / num_rels))
    print('to-1 relations: {}/{} ({})'.format(num_to_1, num_rels, num_to_1 / num_rels))

    to_M_examples = []
    to_1_examples = []
    num_exps = 0
    with open(os.path.join(data_dir, 'dev.triples')) as f:
        for line in f:
            num_exps += 1
            e1, e2, r = line.strip().split('\t')
            if relation2id[r] in to_M_rels:
                to_M_examples.append(line)
            elif relation2id[r] in to_1_rels:
                to_1_examples.append(line)
    num_to_M_exps = len(to_M_examples) + 0.0
    num_to_1_exps = len(to_1_examples) + 0.0
    to_M_ratio = num_to_M_exps / num_exps
    to_1_ratio = num_to_1_exps / num_exps
    print('to-M examples: {}/{} ({})'.format(num_to_M_exps, num_exps, to_M_ratio))
    print('to-1 examples: {}/{} ({})'.format(num_to_1_exps, num_exps, to_1_ratio))

    return to_M_rels, to_1_rels, (to_M_ratio, to_1_ratio)


def load_configs(args, config_path):
    with open(config_path) as f:
        print('loading configuration file {}'.format(config_path))
        for line in f:
            if not '=' in line:
                continue
            arg_name, arg_value = line.strip().split('=')
            if arg_value.startswith('"') and arg_value.endswith('"'):
                arg_value = arg_value[1:-1]
            if hasattr(args, arg_name):
                print('{} = {}'.format(arg_name, arg_value))
                arg_value2 = getattr(args, arg_name)
                if type(arg_value2) is str:
                    setattr(args, arg_name, arg_value)
                elif type(arg_value2) is bool:
                    if arg_value == 'True':
                        setattr(args, arg_name, True)
                    elif arg_value == 'False':
                        setattr(args, arg_name, False)
                    else:
                        raise ValueError('Unrecognized boolean value description: {}'.format(arg_value))
                elif type(arg_value2) is int:
                    setattr(args, arg_name, int(arg_value))
                elif type(arg_value2) is float:
                    setattr(args, arg_name, float(arg_value))
                else:
                    raise ValueError('Unrecognized attribute type: {}: {}'.format(arg_name, type(arg_value2)))
            else:
                raise ValueError('Unrecognized argument: {}'.format(arg_name))
    return args
