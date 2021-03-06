# -*- coding: utf-8 -*-
"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Knowledge Graph Environment.
"""

import collections
import os
import pickle

import torch
import torch.nn as nn

from src.data_utils import load_index
from src.data_utils import NO_OP_ENTITY_ID, NO_OP_RELATION_ID
from src.data_utils import DUMMY_ENTITY_ID, DUMMY_RELATION_ID
from src.data_utils import START_RELATION_ID
import src.utils.ops as ops
from src.utils.ops import int_var_cuda, var_cuda
# from src.utils.seed_sort import sort_idx_by_pr
import numpy as np

CUTOFF = True


class KnowledgeGraph(nn.Module):
    """
    The discrete knowledge graph is stored with an adjacency list.
    """

    def __init__(self, args):
        super(KnowledgeGraph, self).__init__()

        self.use_abstract_graph = args.use_abstract_graph
        print("KnowledgeGraph use_abstract_graph:", self.use_abstract_graph)

        self.entity2id, self.id2entity = {}, {}
        self.relation2id, self.id2relation = {}, {}
        self.type2id, self.id2type = {}, {}
        self.entity2typeid = {}
        self.adj_list = None
        self.adj_list_e2t = None
        self.bandwidth = args.bandwidth
        print("=====>>>> KnowledgeGraph bandwidth:", self.bandwidth, "CUTOFF:", CUTOFF)
        self.args = args

        self.action_space = None
        self.action_space_abs = None
        self.action_space_e2t = None
        self.action_space_buckets = None
        self.unique_r_space = None

        self.train_subjects = None
        self.train_objects = None
        self.dev_subjects = None
        self.dev_objects = None
        self.all_subjects = None
        self.all_objects = None
        self.train_subject_vectors = None
        self.train_object_vectors = None
        self.dev_subject_vectors = None
        self.dev_object_vectors = None
        self.all_subject_vectors = None
        self.all_object_vectors = None

        print('** Create {} knowledge graph **'.format(args.model))
        self.load_graph_data(args.data_dir)
        if self.use_abstract_graph:
            self.load_all_answers_with_abs(args.data_dir)
        else:
            self.load_all_answers(args.data_dir)

        # Define NN Modules
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.emb_dropout_rate = args.emb_dropout_rate
        self.num_graph_convolution_layers = args.num_graph_convolution_layers
        self.entity_embeddings = None
        self.entity_abs_embeddings = None
        self.relation_abs_embeddings = None
        self.relation_embeddings = None
        self.entity_img_embeddings = None
        self.relation_img_embeddings = None
        self.EDropout = None
        self.RDropout = None

        self.define_modules()
        self.initialize_modules()

    def load_graph_data(self, data_dir):
        # Load indices
        self.entity2id, self.id2entity = load_index(os.path.join(data_dir, 'entity2id.txt'))
        print('Sanity check: {} entities loaded'.format(len(self.entity2id)))
        self.type2id, self.id2type = load_index(os.path.join(data_dir, 'type2id.txt'))

        print('Sanity check: {} types loaded'.format(len(self.type2id)))
        with open(os.path.join(data_dir, 'entity2typeid.pkl'), 'rb') as f:
            print("loading entity2typeid.pkl")
            self.entity2typeid = pickle.load(f)
        self.relation2id, self.id2relation = load_index(os.path.join(data_dir, 'relation2id.txt'))
        print('Sanity check: {} relations loaded'.format(len(self.relation2id)))

        # Load graph structures
        if self.args.model.startswith('point'):
            # Base graph structure used for training and test
            adj_list_path = os.path.join(data_dir, 'adj_list.pkl')

            with open(adj_list_path, 'rb') as f:
                self.adj_list = pickle.load(f)
            if self.use_abstract_graph:
                adj_list_abs_path = os.path.join(data_dir, 'adj_list_abs.pkl')
                with open(adj_list_abs_path, 'rb') as fl:
                    print("load adj_list_abs_path")
                    self.adj_list_abs = pickle.load(fl)
                print("===>call self.vectorize_action_space_with_abs(data_dir)")

                adj_list_e2t_path = os.path.join(data_dir, 'adj_list_e2t.pkl')
                with open(adj_list_e2t_path, 'rb') as fl:
                    print("load adj_list_e2t_path")
                    self.adj_list_e2t = pickle.load(fl)
                print("===>call self.vectorize_action_space_with_e2t(data_dir)")
                self.vectorize_action_space_with_abs(data_dir)
            else:
                self.vectorize_action_space(data_dir)

    def vectorize_action_space(self, data_dir):
        """
        Pre-process and numericalize the knowledge graph structure.
        """

        def load_page_rank_scores(input_path):
            pgrk_scores = collections.defaultdict(float)
            with open(input_path) as f:
                for line in f:
                    e, score = line.strip().split(':')
                    e_id = self.entity2id[e.strip()]
                    score = float(score)
                    pgrk_scores[e_id] = score
            return pgrk_scores

        # Sanity check
        num_facts = 0
        out_degrees = collections.defaultdict(int)
        for e1 in self.adj_list:
            for r in self.adj_list[e1]:
                num_facts += len(self.adj_list[e1][r])
                out_degrees[e1] += len(self.adj_list[e1][r])
        print("Sanity check: maximum out degree: {}".format(max(out_degrees.values())))
        print('Sanity check: {} facts in knowledge graph'.format(num_facts))

        # load page rank scores
        page_rank_scores = load_page_rank_scores(os.path.join(data_dir, 'raw.pgrk'))

        def get_action_space(e1):
            action_space = []
            if e1 in self.adj_list:
                for r in self.adj_list[e1]:
                    targets = self.adj_list[e1][r]
                    for e2 in targets:
                        action_space.append((r, e2))

                if len(action_space) + 1 >= self.bandwidth:
                    # Base graph pruning
                    sorted_action_space = \
                        sorted(action_space, key=lambda x: page_rank_scores[x[1]], reverse=True)
                    action_space = sorted_action_space[:self.bandwidth]
            action_space.insert(0, (NO_OP_RELATION_ID, e1))
            return action_space

        def get_unique_r_space(e1):
            if e1 in self.adj_list:
                return list(self.adj_list[e1].keys())
            else:
                return []

        def vectorize_action_space(action_space_list, action_space_size):
            bucket_size = len(action_space_list)
            r_space = torch.zeros(bucket_size, action_space_size) + self.dummy_r
            e_space = torch.zeros(bucket_size, action_space_size) + self.dummy_e
            action_mask = torch.zeros(bucket_size, action_space_size)
            for i, action_space in enumerate(action_space_list):
                for j, (r, e) in enumerate(action_space):
                    r_space[i, j] = r
                    e_space[i, j] = e
                    action_mask[i, j] = 1
            return (int_var_cuda(r_space), int_var_cuda(e_space)), var_cuda(action_mask)

        def vectorize_unique_r_space(unique_r_space_list, unique_r_space_size, volatile):
            bucket_size = len(unique_r_space_list)
            unique_r_space = torch.zeros(bucket_size, unique_r_space_size) + self.dummy_r
            for i, u_r_s in enumerate(unique_r_space_list):
                for j, r in enumerate(u_r_s):
                    unique_r_space[i, j] = r
            return int_var_cuda(unique_r_space)

        if self.args.use_action_space_bucketing:
            """
            Store action spaces in buckets.
            """
            self.action_space_buckets = {}
            action_space_buckets_discrete = collections.defaultdict(list)
            self.entity2bucketid = torch.zeros(self.num_entities, 2).long()
            num_facts_saved_in_action_table = 0
            for e1 in range(self.num_entities):
                action_space = get_action_space(e1)
                key = int(len(action_space) / self.args.bucket_interval) + 1
                self.entity2bucketid[e1, 0] = key
                self.entity2bucketid[e1, 1] = len(action_space_buckets_discrete[key])
                action_space_buckets_discrete[key].append(action_space)
                num_facts_saved_in_action_table += len(action_space)
            print('Sanity check: {} facts saved in action table'.format(
                num_facts_saved_in_action_table - self.num_entities))
            for key in action_space_buckets_discrete:
                print('Vectorizing action spaces bucket {}...'.format(key))
                self.action_space_buckets[key] = vectorize_action_space(
                    action_space_buckets_discrete[key], key * self.args.bucket_interval)
        else:
            action_space_list = []
            max_num_actions = 0
            for e1 in range(self.num_entities):
                action_space = get_action_space(e1)
                action_space_list.append(action_space)
                if len(action_space) > max_num_actions:
                    max_num_actions = len(action_space)
            print('Vectorizing action spaces...')
            self.action_space = vectorize_action_space(action_space_list, max_num_actions)

            if self.args.model.startswith('rule'):
                unique_r_space_list = []
                max_num_unique_rs = 0
                for e1 in sorted(self.adj_list.keys()):
                    unique_r_space = get_unique_r_space(e1)
                    unique_r_space_list.append(unique_r_space)
                    if len(unique_r_space) > max_num_unique_rs:
                        max_num_unique_rs = len(unique_r_space)
                self.unique_r_space = vectorize_unique_r_space(unique_r_space_list, max_num_unique_rs)

    def vectorize_action_space_with_abs(self, data_dir):
        """
        Pre-process and numericalize the knowledge graph structure.
        """
        print("vectorize_action_space_with_abs")

        def load_page_rank_scores(input_path):
            pgrk_scores = collections.defaultdict(float)
            pgrk_abs_scores = collections.defaultdict(float)
            with open(input_path) as f:
                for line in f:
                    e, score = line.strip().split(':')
                    e_id = self.entity2id[e.strip()]
                    t_id = self.get_typeid(e_id)
                    score = float(score)
                    pgrk_scores[e_id] = score
                    pgrk_abs_scores[t_id] += score
            return pgrk_scores, pgrk_abs_scores

        # Sanity check
        num_facts = 0
        num_facts_abs = 0
        num_facts_e2t = 0

        out_degrees = collections.defaultdict(int)
        out_degrees_abs = collections.defaultdict(int)
        out_degrees_e2t = collections.defaultdict(int)

        for e1 in self.adj_list:
            for r in self.adj_list[e1]:
                num_facts += len(self.adj_list[e1][r])
                out_degrees[e1] += len(self.adj_list[e1][r])
                # e1_abs = self.get_typeid(e1)
        for e1_abs in self.adj_list_abs:
            for r in self.adj_list_abs[e1_abs]:
                num_facts_abs += len(self.adj_list_abs[e1_abs][r])
                out_degrees_abs[e1_abs] += len(self.adj_list_abs[e1_abs][r])
        for e1 in self.adj_list_e2t:
            for r in self.adj_list_e2t[e1]:
                num_facts_e2t += len(self.adj_list_e2t[e1][r])
                out_degrees_e2t[e1] += len(self.adj_list_e2t[e1][r])

        print("Sanity check: maximum out degree: {},  max out abs degree:{}, max out type degree:{}".format(
            max(out_degrees.values()),
            max(out_degrees_abs.values()), max(out_degrees_e2t.values())))
        print(
            'Sanity check: {} facts [abs {} facts] [type {} facts] in knowledge graph'.format(num_facts, num_facts_abs,
                                                                                              num_facts_e2t))

        # load page rank scores
        page_rank_scores, page_rank_abs_scores = load_page_rank_scores(os.path.join(data_dir, 'raw.pgrk'))

        # def get_action_space(e1):
        #     action_space = []
        #     action_space_abs = []
        #     if e1 in self.adj_list:
        #         e1_abs = self.get_typeid(e1)
        #         for r in self.adj_list[e1]:
        #             targets = self.adj_list[e1][r]
        #             for e2 in targets:
        #                 action_space.append((r, e2))
        #         for r in self.adj_list_abs[e1_abs]:
        #             targets = self.adj_list_abs[e1_abs][r]
        #             for e2_abs in targets:
        #                 action_space_abs.append((r, e2_abs))
        #         if len(action_space) + 1 >= self.bandwidth:
        #             # Base graph pruning
        #             sorted_action_space = \
        #                 sorted(action_space, key=lambda x: page_rank_scores[x[1]], reverse=True)
        #             action_space = sorted_action_space[:self.bandwidth]
        #         if len(action_space_abs) + 1 >= self.bandwidth:
        #             action_space_abs = sorted(action_space_abs, key=lambda x: page_rank_abs_scores[x[1]], reverse=True)[
        #                                :self.bandwidth]
        #
        #     action_space.insert(0, (NO_OP_RELATION_ID, e1))
        #     action_space_abs.insert(0, (NO_OP_RELATION_ID, self.get_typeid(e1)))
        #     return action_space, action_space_abs


        def get_action_space(e1):
            action_space = []
            if e1 in self.adj_list:
                for r in self.adj_list[e1]:
                    targets = self.adj_list[e1][r]
                    for e2 in targets:
                        action_space.append((r, e2))
                if CUTOFF and len(action_space) + 1 >= self.bandwidth:
                    # Base graph pruning
                    sorted_action_space = \
                        sorted(action_space, key=lambda x: page_rank_scores[x[1]], reverse=True)
                    action_space = sorted_action_space[:self.bandwidth]
            action_space.insert(0, (NO_OP_RELATION_ID, e1))
            return action_space

        def get_action_space_abs(e1_abs):
            action_space_abs = []
            if e1_abs in self.adj_list_abs:
                for r in self.adj_list_abs[e1_abs]:
                    targets = self.adj_list_abs[e1_abs][r]
                    for e2_abs in targets:
                        action_space_abs.append((r, e2_abs))
                if CUTOFF and len(action_space_abs) + 1 >= self.bandwidth:
                    action_space_abs = sorted(action_space_abs, key=lambda x: page_rank_abs_scores[x[1]], reverse=True)[
                                       :self.bandwidth]
            action_space_abs.insert(0, (NO_OP_RELATION_ID, e1_abs))
            return action_space_abs

        def get_action_space_e2t(e1):
            action_space_abs = []
            if e1 in self.adj_list_e2t:
                for r in self.adj_list_e2t[e1]:
                    targets = self.adj_list_e2t[e1][r]
                    for type in targets:
                        action_space_abs.append((r, type))
                if CUTOFF and len(action_space_abs) + 1 >= self.bandwidth:
                    action_space_abs = sorted(action_space_abs, key=lambda x: page_rank_abs_scores[x[1]], reverse=True)[
                                       :self.bandwidth]
            action_space_abs.insert(0, (NO_OP_RELATION_ID, e1_abs))
            return action_space_abs

        def get_two_action_space(e1):
            action_space = []
            action_space_e2t = []
            if e1 in self.adj_list and e1 in self.adj_list_e2t:
                for r in self.adj_list[e1]:
                    targets = self.adj_list[e1][r]
                    abs_targets = self.adj_list_e2t[e1][r]
                    for _i in range(len(targets)):
                        assert self.entity2typeid[targets[_i]] == abs_targets[_i]
                    for _ in range(len(targets)):
                        action_space.append((r, targets[_]))
                        action_space_e2t.append((r, abs_targets[_]))

                if CUTOFF and len(action_space) + 1 >= self.bandwidth:  # 排序并去重！！！去重！！！
                    action_space = np.asarray(action_space, dtype=np.int32)
                    action_space_e2t = np.asarray(action_space_e2t, dtype=np.int32)
                    # Base graph pruning
                    #seed = np.random.randint(0, self.num_entities)
                    array = np.array([page_rank_scores[_[1]]
                             for i, _ in enumerate(action_space)])
          
                    #idx = sort_idx_by_pr(array, seed)
                    idx = np.argsort(array)[::-1] #从大到小的index

                    sorted_action_space = [(int(_[0]), int(_[1])) for _ in action_space[idx]]
                    sorted_e2t_action_space = [(int(_[0]), int(_[1])) for _ in action_space_e2t[idx]]

                    # sorted_action_space = \
                    #     sorted(
                    #         action_space, key=lambda x: page_rank_scores[x[1]], reverse=True)
                    # sorted_abs_action_space = \
                    #     sorted(
                    #         action_space_abs, key=lambda x: page_rank_scores[x[1]], reverse=True)

                    action_space = sorted_action_space[:self.bandwidth]
                    action_space_e2t = sorted_e2t_action_space[:self.bandwidth]
                    # if e1 == 48:
                        # print([(_[0], _[1], self.entity2typeid[_[1]]) for _ in action_space])
                        # print("")
                        # print([(_[0], _[1]) for _ in action_space_e2t])
                        # print("")
                        # abs_type_set = set( [_[1] for _ in action_space_e2t] )
                        # real_type_set = set( [ self.entity2typeid[_[1]] for _ in action_space ] )
                        # print( (real_type_set&abs_type_set) == real_type_set )
                        # print("real-abs:",  real_type_set - abs_type_set )
                        # print("abs-:",  abs_type_set - real_type_set)
                # action_space_e2t = list(set(action_space_e2t))
                action_space.insert(0, (NO_OP_RELATION_ID, e1))
                action_space_e2t.insert(
                    0, (NO_OP_RELATION_ID, self.entity2typeid[e1]))
                if e1 == 48:
                    print(action_space)
                    print(action_space_e2t)
            return action_space, action_space_e2t

        def get_unique_r_space(e1, adj_list=self.adj_list):
            if e1 in adj_list:
                return list(adj_list[e1].keys())
            else:
                return []

        def vectorize_action_space(action_space_list, action_space_size):
            bucket_size = len(action_space_list)
            r_space = torch.zeros(bucket_size, action_space_size) + self.dummy_r
            e_space = torch.zeros(bucket_size, action_space_size) + self.dummy_e
            action_mask = torch.zeros(bucket_size, action_space_size)
            for i, action_space in enumerate(action_space_list):
                for j, (r, e) in enumerate(action_space):
                    r_space[i, j] = r
                    e_space[i, j] = e
                    action_mask[i, j] = 1
            return (int_var_cuda(r_space), int_var_cuda(e_space)), var_cuda(action_mask)

        def vectorize_unique_r_space(unique_r_space_list, unique_r_space_size, volatile):
            bucket_size = len(unique_r_space_list)
            unique_r_space = torch.zeros(bucket_size, unique_r_space_size) + self.dummy_r
            for i, u_r_s in enumerate(unique_r_space_list):
                for j, r in enumerate(u_r_s):
                    unique_r_space[i, j] = r
            return int_var_cuda(unique_r_space)

        if self.args.use_action_space_bucketing:
            """
            Store action spaces in buckets.
            """
            self.action_space_buckets = {}
            self.action_space_abs_buckets = {}
            action_space_buckets_discrete = collections.defaultdict(list)
            action_space_abs_buckets_discrete = collections.defaultdict(list)
            self.entity2bucketid = torch.zeros(self.num_entities, 2).long()
            self.entityabs2bucketid = torch.zeros(self.num_entities_type, 2).long()
            num_facts_saved_in_action_table = 0
            num_facts_abs_saved_in_action_table = 0
            for e1 in range(self.num_entities):  # TODO:CHECK这里可鞥呢是值遍历e1 in range(num_entities_types)就行了
                # e1_abs = self.get_typeid(e1)
                action_space = get_action_space(e1)
                key = int(len(action_space) / self.args.bucket_interval) + 1
                # key_abs = int(len(action_space_abs) / self.args.bucket_interval) + 1
                self.entity2bucketid[e1, 0] = key
                # self.entityabs2bucketid[e1_abs, 0] = key_abs
                self.entity2bucketid[e1, 1] = len(action_space_buckets_discrete[key])
                # self.entityabs2bucketid[e1_abs, 1] = len(action_space_abs_buckets_discrete[key_abs])
                action_space_buckets_discrete[key].append(action_space)
                # action_space_abs_buckets_discrete[key_abs].append(action_space_abs)
                num_facts_saved_in_action_table += len(action_space)
                # num_facts_abs_saved_in_action_table += len(action_space_abs)

            for e1_abs in range(self.num_entities_type):
                action_space_abs = get_action_space_abs(e1_abs)  # [出边(r,e2)的个数+1]， 因为起始塞了一个
                key_abs = int(len(action_space_abs) / self.args.bucket_interval) + 1
                self.entityabs2bucketid[e1_abs, 0] = key_abs  # bucket_key
                self.entityabs2bucketid[e1_abs, 1] = len(action_space_abs_buckets_discrete[key_abs])
                action_space_abs_buckets_discrete[key_abs].append(action_space_abs)
                num_facts_abs_saved_in_action_table += len(action_space_abs)

            print('Sanity check: {} facts saved [abs {}] in action table'.format(
                num_facts_saved_in_action_table - self.num_entities,
                num_facts_abs_saved_in_action_table - self.num_entities_type))
            for key in action_space_buckets_discrete:
                print('Vectorizing action spaces bucket {}...'.format(key))
                self.action_space_buckets[key] = vectorize_action_space(
                    action_space_buckets_discrete[key], key * self.args.bucket_interval)
            for key in action_space_abs_buckets_discrete:
                print('Vectorizing action spaces bucket abs {}...'.format(key))
                self.action_space_abs_buckets[key] = vectorize_action_space(
                    action_space_abs_buckets_discrete[key], key * self.args.bucket_interval)  # TODO:CHECK 这里这个函数是不是可以共用
        else:
            action_space_list = []
            action_space_abs_list = []
            action_space_e2t_list = []

            max_num_actions = 0
            max_num_actions_abs = 0
            max_num_actions_e2t = 0

            # for e1 in range(self.num_entities):

            #     action_space = get_action_space(e1)

            #     action_space_list.append(action_space)
            #     if len(action_space) > max_num_actions:
            #         max_num_actions = len(action_space)

            for e1_abs in range(self.num_entities_type):
                e1_abs = self.get_typeid(e1_abs)
                action_space_abs = get_action_space_abs(e1_abs)
                action_space_abs_list.append(action_space_abs)
                if len(action_space_abs) > max_num_actions_abs:
                    max_num_actions_abs = len(action_space_abs)

            # for e1 in range(self.num_entities):
            #     action_space_e2t = get_action_space_e2t(e1)
            #     action_space_e2t_list.append(action_space_e2t)
            #     if len(action_space_e2t) > max_num_actions_e2t:
            #         max_num_actions_e2t = len(action_space_e2t)

            for e1 in range(self.num_entities):
                action_space, action_space_e2t = get_two_action_space(e1)
                action_space_list.append(action_space)
                action_space_e2t_list.append(action_space_e2t)
                if len(action_space) > max_num_actions:
                    max_num_actions = len(action_space)
                if len(action_space_e2t) > max_num_actions_e2t:
                    max_num_actions_e2t = len(action_space_e2t)

            print('Vectorizing action spaces...max_num_actions {} max_num_actions_abs {} max_num_actions_e2t {},'.format(
                max_num_actions, max_num_actions_abs, max_num_actions_e2t))
            self.action_space = vectorize_action_space(action_space_list, max_num_actions)
            self.action_space_abs = vectorize_action_space(action_space_abs_list, max_num_actions_abs)
            self.action_space_e2t = vectorize_action_space(action_space_e2t_list, max_num_actions_e2t)

            if self.args.model.startswith('rule'):
                raise NotImplementedError
                unique_r_space_list = []
                max_num_unique_rs = 0
                for e1 in sorted(self.adj_list.keys()):
                    unique_r_space = get_unique_r_space(e1, self.adj_list)
                    unique_r_space_list.append(unique_r_space)
                    if len(unique_r_space) > max_num_unique_rs:
                        max_num_unique_rs = len(unique_r_space)
                self.unique_r_space = vectorize_unique_r_space(unique_r_space_list, max_num_unique_rs)

                unique_r_space_list_abs = []
                max_num_unique_rs_abs = 0
                for e1 in sorted(self.adj_list_abs.keys()):
                    unique_r_space_abs = get_unique_r_space(e1, self.adj_list_abs)
                    unique_r_space_list_abs.append(unique_r_space)
                    if len(unique_r_space_abs) > max_num_unique_rs_abs:
                        max_num_unique_rs_abs = len(unique_r_space_abs)
                self.unique_r_space_abs = vectorize_unique_r_space(unique_r_space_list_abs,
                                                                   max_num_unique_rs_abs)  # TODO:CHECK 这里是不是可以共用

    def load_all_answers(self, data_dir, add_reversed_edges=False):
        def add_subject(e1, e2, r, d):
            if not e2 in d:
                d[e2] = {}
            if not r in d[e2]:
                d[e2][r] = set()
            d[e2][r].add(e1)

        def add_object(e1, e2, r, d):
            if not e1 in d:
                d[e1] = {}
            if not r in d[e1]:
                d[e1][r] = set()
            d[e1][r].add(e2)

        # store subjects for all (rel, object) queries and
        # objects for all (subject, rel) queries
        train_subjects, train_objects = {}, {}
        dev_subjects, dev_objects = {}, {}
        all_subjects, all_objects = {}, {}
        # include dummy examples
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, train_subjects)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, dev_subjects)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, all_subjects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, train_objects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, dev_objects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, all_objects)
        for file_name in ['raw.kb', 'train.triples', 'dev.triples', 'test.triples']:
            if 'NELL' in self.args.data_dir and self.args.test and file_name == 'train.triples':
                continue
            with open(os.path.join(data_dir, file_name)) as f:
                for line in f:
                    e1, e2, r = line.strip().split()
                    e1, e2, r = self.triple2ids((e1, e2, r))
                    if file_name in ['raw.kb', 'train.triples']:
                        add_subject(e1, e2, r, train_subjects)
                        add_object(e1, e2, r, train_objects)
                        if add_reversed_edges:
                            add_subject(e2, e1, self.get_inv_relation_id(r), train_subjects)
                            add_object(e2, e1, self.get_inv_relation_id(r), train_objects)
                    if file_name in ['raw.kb', 'train.triples', 'dev.triples']:
                        add_subject(e1, e2, r, dev_subjects)
                        add_object(e1, e2, r, dev_objects)
                        if add_reversed_edges:
                            add_subject(e2, e1, self.get_inv_relation_id(r), dev_subjects)
                            add_object(e2, e1, self.get_inv_relation_id(r), dev_objects)
                    add_subject(e1, e2, r, all_subjects)
                    add_object(e1, e2, r, all_objects)
                    if add_reversed_edges:
                        add_subject(e2, e1, self.get_inv_relation_id(r), all_subjects)
                        add_object(e2, e1, self.get_inv_relation_id(r), all_objects)
        self.train_subjects = train_subjects
        self.train_objects = train_objects
        self.dev_subjects = dev_subjects
        self.dev_objects = dev_objects
        self.all_subjects = all_subjects
        self.all_objects = all_objects

        # change the answer set into a variable
        def answers_to_var(d_l):
            d_v = collections.defaultdict(collections.defaultdict)
            for x in d_l:
                for y in d_l[x]:
                    v = torch.LongTensor(list(d_l[x][y])).unsqueeze(1)
                    d_v[x][y] = int_var_cuda(v)
            return d_v

        self.train_subject_vectors = answers_to_var(train_subjects)
        self.train_object_vectors = answers_to_var(train_objects)
        self.dev_subject_vectors = answers_to_var(dev_subjects)
        self.dev_object_vectors = answers_to_var(dev_objects)
        self.all_subject_vectors = answers_to_var(all_subjects)
        self.all_object_vectors = answers_to_var(all_objects)

    def load_all_answers_with_abs(self, data_dir, add_reversed_edges=False):
        print("load_all_answers_with_abs")

        def add_subject(e1, e2, r, d):
            if not e2 in d:
                d[e2] = {}
            if not r in d[e2]:
                d[e2][r] = set()
            d[e2][r].add(e1)

        def add_object(e1, e2, r, d):
            if not e1 in d:
                d[e1] = {}
            if not r in d[e1]:
                d[e1][r] = set()
            d[e1][r].add(e2)

        # store subjects for all (rel, object) queries and
        # objects for all (subject, rel) queries
        train_subjects, train_objects = {}, {}
        train_subjects_abs, train_objects_abs = {}, {}
        dev_subjects, dev_objects = {}, {}
        dev_subjects_abs, dev_objects_abs = {}, {}
        all_subjects, all_objects = {}, {}
        all_subjects_abs, all_objects_abs = {}, {}
        # include dummy examples
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, train_subjects)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, dev_subjects)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, all_subjects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, train_objects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, dev_objects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, all_objects)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, train_subjects_abs)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, dev_subjects_abs)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, all_subjects_abs)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, train_objects_abs)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, dev_objects_abs)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, all_objects_abs)
        for file_name in ['raw.kb', 'train.triples', 'dev.triples', 'test.triples']:
            if 'NELL' in self.args.data_dir and self.args.test and file_name == 'train.triples':
                continue
            with open(os.path.join(data_dir, file_name)) as f:
                for line in f:
                    e1, e2, r = line.strip().split()
                    e1, e2, r = self.triple2ids((e1, e2, r))
                    e1_abs, e2_abs, r_abs = self.entity2typeid[e1], self.entity2typeid[e2], r
                    if file_name in ['raw.kb', 'train.triples']:
                        add_subject(e1, e2, r, train_subjects)
                        add_subject(e1_abs, e2_abs, r_abs, train_subjects_abs)
                        add_object(e1, e2, r, train_objects)
                        add_object(e1_abs, e2_abs, r_abs, train_objects_abs)
                        if add_reversed_edges:
                            add_subject(e2, e1, self.get_inv_relation_id(r), train_subjects)
                            add_subject(e2_abs, e1_abs, self.get_inv_relation_id(r_abs),
                                        train_subjects_abs)  # TODO:CHECKget_inv_relation_id
                            add_object(e2, e1, self.get_inv_relation_id(r), train_objects)
                            add_object(e2_abs, e1_abs, self.get_inv_relation_id(r_abs),
                                       train_objects_abs)  # TODO:CHECKget_inv_relation_id
                    if file_name in ['raw.kb', 'train.triples', 'dev.triples']:
                        add_subject(e1, e2, r, dev_subjects)
                        add_subject(e1_abs, e2_abs, r_abs, dev_subjects_abs)
                        add_object(e1, e2, r, dev_objects)
                        add_object(e1_abs, e2_abs, r_abs, dev_objects_abs)
                        if add_reversed_edges:
                            add_subject(e2, e1, self.get_inv_relation_id(r), dev_subjects)
                            add_subject(e2_abs, e1_abs, self.get_inv_relation_id(r_abs), dev_subjects_abs)
                            add_object(e2, e1, self.get_inv_relation_id(r), dev_objects)
                            add_object(e2_abs, e1_abs, self.get_inv_relation_id(r_abs), dev_objects_abs)
                    add_subject(e1, e2, r, all_subjects)
                    add_subject(e1_abs, e2_abs, r_abs, all_subjects_abs)
                    add_object(e1, e2, r, all_objects)
                    add_object(e1_abs, e2_abs, r_abs, all_objects_abs)
                    if add_reversed_edges:
                        add_subject(e2, e1, self.get_inv_relation_id(r), all_subjects)
                        add_subject(e2_abs, e1_abs, self.get_inv_relation_id(r_abs), all_subjects_abs)
                        add_object(e2, e1, self.get_inv_relation_id(r), all_objects)
                        add_object(e2_abs, e1_abs, self.get_inv_relation_id(r_abs), all_objects_abs)
        self.train_subjects = train_subjects
        self.train_subjects_abs = train_subjects_abs
        self.train_objects = train_objects
        self.train_objects_abs = train_objects_abs
        self.dev_subjects = dev_subjects
        self.dev_subjects_abs = dev_subjects_abs
        self.dev_objects = dev_objects
        self.dev_objects_abs = dev_objects_abs
        self.all_subjects = all_subjects
        self.all_subjects_abs = all_subjects_abs
        self.all_objects = all_objects
        self.all_objects_abs = all_objects_abs

        # change the answer set into a variable
        def answers_to_var(d_l):
            d_v = collections.defaultdict(collections.defaultdict)
            for x in d_l:
                for y in d_l[x]:
                    v = torch.LongTensor(list(d_l[x][y])).unsqueeze(1)
                    d_v[x][y] = int_var_cuda(v)
            return d_v

        self.train_subject_vectors = answers_to_var(train_subjects)
        self.train_subject_vectors_abs = answers_to_var(train_subjects_abs)
        self.train_object_vectors = answers_to_var(train_objects)
        self.train_object_vectors_abs = answers_to_var(train_objects_abs)
        self.dev_subject_vectors = answers_to_var(dev_subjects)
        self.dev_subject_vectors_abs = answers_to_var(dev_subjects_abs)
        self.dev_object_vectors = answers_to_var(dev_objects)
        self.dev_object_vectors_abs = answers_to_var(dev_objects_abs)
        self.all_subject_vectors = answers_to_var(all_subjects)
        self.all_subject_vectors_abs = answers_to_var(all_subjects_abs)
        self.all_object_vectors = answers_to_var(all_objects)
        self.all_object_vectors_abs = answers_to_var(all_objects_abs)

    def load_fuzzy_facts(self):
        # extend current adjacency list with fuzzy facts
        dev_path = os.path.join(self.args.data_dir, 'dev.triples')
        test_path = os.path.join(self.args.data_dir, 'test.triples')
        with open(dev_path) as f:
            dev_triples = [l.strip() for l in f.readlines()]
        with open(test_path) as f:
            test_triples = [l.strip() for l in f.readlines()]
        removed_triples = set(dev_triples + test_triples)
        theta = 0.5
        fuzzy_fact_path = os.path.join(self.args.data_dir, 'train.fuzzy.triples')
        count = 0
        with open(fuzzy_fact_path) as f:
            for line in f:
                e1, e2, r, score = line.strip().split()
                score = float(score)
                if score < theta:
                    continue
                print(line)
                if '{}\t{}\t{}'.format(e1, e2, r) in removed_triples:
                    continue
                e1_id = self.entity2id[e1]
                e2_id = self.entity2id[e2]
                r_id = self.relation2id[r]
                if not r_id in self.adj_list[e1_id]:
                    self.adj_list[e1_id][r_id] = set()
                if not e2_id in self.adj_list[e1_id][r_id]:
                    self.adj_list[e1_id][r_id].add(e2_id)
                    count += 1
                    if count > 0 and count % 1000 == 0:
                        print('{} fuzzy facts added'.format(count))

        self.vectorize_action_space(self.args.data_dir)

    def get_typeid(self, e):
        # if type == str:
        #     print("e:", e)
        # e = int(e)
        return self.entity2typeid[e]

    def get_inv_relation_id(self, r_id):
        return r_id + 1

    def get_all_entity_embeddings(self):
        return self.EDropout(self.entity_embeddings.weight)

    def get_entity_embeddings(self, e):
        return self.EDropout(self.entity_embeddings(e))

    def get_entity_abs_embeddings(self, e):
        return self.EDropout(self.entity_abs_embeddings(e))

    def get_all_relation_embeddings(self):
        return self.RDropout(self.relation_embeddings.weight)

    def get_relation_embeddings(self, r):
        return self.RDropout(self.relation_embeddings(r))

    def get_relation_abs_embeddings(self, r):
        return self.RDropout(self.relation_abs_embeddings(r))

    def get_all_entity_img_embeddings(self):
        return self.EDropout(self.entity_img_embeddings.weight)

    def get_entity_img_embeddings(self, e):
        return self.EDropout(self.entity_img_embeddings(e))

    def get_relation_img_embeddings(self, r):
        return self.RDropout(self.relation_img_embeddings(r))

    def virtual_step(self, e_set, r):
        """
        Given a set of entities (e_set), find the set of entities (e_set_out) which has at least one incoming edge
        labeled r and the source entity is in e_set.
        """
        batch_size = len(e_set)
        e_set_1D = e_set.view(-1)
        r_space = self.action_space[0][0][e_set_1D]
        e_space = self.action_space[0][1][e_set_1D]
        e_space = (r_space.view(batch_size, -1) == r.unsqueeze(1)).long() * e_space.view(batch_size, -1)
        e_set_out = []
        for i in range(len(e_space)):
            e_set_out_b = var_cuda(unique(e_space[i].data))
            e_set_out.append(e_set_out_b.unsqueeze(0))
        e_set_out = ops.pad_and_cat(e_set_out, padding_value=self.dummy_e)
        return e_set_out

    def id2triples(self, triple):
        e1, e2, r = triple
        return self.id2entity[e1], self.id2entity[e2], self.id2relation[r]

    def triple2ids(self, triple):
        e1, e2, r = triple
        return self.entity2id[e1], self.entity2id[e2], self.relation2id[r]

    def define_modules(self):
        if not self.args.relation_only:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.entity_dim)

            if self.args.model == 'complex':
                self.entity_img_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
            self.EDropout = nn.Dropout(self.emb_dropout_rate)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
        if self.use_abstract_graph:
            if not self.args.relation_only:
                self.entity_abs_embeddings = nn.Embedding(self.num_entities_type, self.entity_dim)
            self.relation_abs_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
        if self.args.model == 'complex':
            self.relation_img_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
        self.RDropout = nn.Dropout(self.emb_dropout_rate)

    def initialize_modules(self):
        if not self.args.relation_only:
            nn.init.xavier_normal_(self.entity_embeddings.weight)
            if self.use_abstract_graph:
                nn.init.xavier_normal_(self.entity_abs_embeddings.weight)
                # nn.init.xavier_normal_(self.relation_abs_embeddings.weight) #CHECK here why not error...
        if self.use_abstract_graph:
            nn.init.xavier_normal_(self.relation_abs_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

    @property
    def num_entities(self):
        return len(self.entity2id)

    @property
    def num_entities_type(self):
        return len(self.type2id)

    @property
    def num_relations(self):
        return len(self.relation2id)

    @property
    def self_edge(self):
        return NO_OP_RELATION_ID

    @property
    def self_e(self):
        return NO_OP_ENTITY_ID

    @property
    def dummy_r(self):
        return DUMMY_RELATION_ID

    @property
    def dummy_e(self):
        return DUMMY_ENTITY_ID

    @property
    def dummy_start_r(self):
        return START_RELATION_ID
