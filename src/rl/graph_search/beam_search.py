# -*- coding: utf-8 -*-
"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Beam search on the graph.
"""

import torch
import time
import json
import pickle

import src.utils.ops as ops
from src.utils.ops import unique_max, var_cuda, zeros_var_cuda, int_var_cuda, int_fill_var_cuda, var_to_numpy
import numpy as np


REAL_ALL_PATHS = []

def beam_search(pn, e_s, q, e_t, kg, num_steps, beam_size, return_path_components=False, return_merge_scores=None):
    """
    Beam search from source.

    :param pn: Policy network.
    :param e_s: (Variable:batch) source entity indices.
    :param q: (Variable:batch) query relation indices.
    :param e_t: (Variable:batch) target entity indices.
    :param kg: Knowledge graph environment.
    :param num_steps: Number of search steps.
    :param beam_size: Beam size used in search.
    :param return_path_components: If set, return all path components at the end of search.
    """
    assert (num_steps >= 1)
    batch_size = len(e_s)

    def top_k_action(log_action_dist, action_space, return_merge_scores=None):
        """
        Get top k actions.
            - k = beam_size if the beam size is smaller than or equal to the beam action space size
            - k = beam_action_space_size otherwise
        :param log_action_dist: [batch_size*k, action_space_size]
        :param action_space (r_space, e_space):
            r_space: [batch_size*k, action_space_size]
            e_space: [batch_size*k, action_space_size]
        :return:
            (next_r, next_e), action_prob, action_offset: [batch_size*new_k]
        """
        full_size = len(log_action_dist)
        assert (full_size % batch_size == 0)
        last_k = int(full_size / batch_size)

        (r_space, e_space), _ = action_space
        action_space_size = r_space.size()[1]
        # => [batch_size, k'*action_space_size]
        log_action_dist = log_action_dist.view(batch_size, -1)
        beam_action_space_size = log_action_dist.size()[1]
        k = min(beam_size, beam_action_space_size)

        if return_merge_scores is not None:
            if return_merge_scores == 'sum':
                reduce_method = torch.sum
            elif return_merge_scores == 'mean':
                reduce_method = torch.mean
            else:
                reduce_method = None

            all_action_ind = torch.LongTensor([range(beam_action_space_size) for _ in range(len(log_action_dist))]).cuda(device=0)

            all_next_r = ops.batch_lookup(r_space.view(batch_size, -1), all_action_ind)
            all_next_e = ops.batch_lookup(e_space.view(batch_size, -1), all_action_ind)

            real_log_action_prob, real_next_e, real_action_ind, real_next_r = ops.merge_same(log_action_dist, all_next_e, all_next_r, method=reduce_method)

            next_e_list, next_r_list, action_ind_list, log_action_prob_list = [], [], [], []
            for i in range(batch_size):
                k_prime = min(len(real_log_action_prob[i]), k)
                top_log_prob, top_ind = torch.topk(real_log_action_prob[i], k_prime)
                top_next_e, top_next_r, top_ind = real_next_e[i][top_ind], real_next_r[i][top_ind], real_action_ind[i][top_ind]
                next_e_list.append(top_next_e.unsqueeze(0))
                next_r_list.append(top_next_r.unsqueeze(0))
                action_ind_list.append(top_ind.unsqueeze(0))
                log_action_prob_list.append(top_log_prob.unsqueeze(0))

            next_r = ops.pad_and_cat(next_r_list, padding_value=kg.dummy_r).view(-1)
            next_e = ops.pad_and_cat(next_e_list, padding_value=kg.dummy_e).view(-1)
            log_action_prob = ops.pad_and_cat(log_action_prob_list, padding_value=0.0).view(-1)
            action_ind = ops.pad_and_cat(action_ind_list, padding_value=-1).view(-1)
        else:
            log_action_prob, action_ind = torch.topk(log_action_dist, k)
            next_r = ops.batch_lookup(r_space.view(batch_size, -1), action_ind).view(-1)
            next_e = ops.batch_lookup(e_space.view(batch_size, -1), action_ind).view(-1)

        # [batch_size, k] => [batch_size*k]
        log_action_prob = log_action_prob.view(-1)
        # *** compute parent offset
        # [batch_size, k]
        action_beam_offset = action_ind / action_space_size
        # [batch_size, 1]
        action_batch_offset = int_var_cuda(torch.arange(batch_size) * last_k).unsqueeze(1)
        # [batch_size, k] => [batch_size*k]
        action_offset = (action_batch_offset + action_beam_offset).view(-1)
        return (next_r, next_e), log_action_prob, action_offset

    def top_k_answer_unique(log_action_dist, action_space):
        """
        Get top k unique entities
            - k = beam_size if the beam size is smaller than or equal to the beam action space size
            - k = beam_action_space_size otherwise
        :param log_action_dist: [batch_size*beam_size, action_space_size] 概率
        :param action_space (r_space, e_space): 实体
            r_space: [batch_size*beam_size, action_space_size]
            e_space: [batch_size*beam_size, action_space_size]
        :return:
            (next_r, next_e), action_prob, action_offset: [batch_size*k]
        """
        full_size = len(log_action_dist)
        assert (full_size % batch_size == 0)
        last_k = int(full_size / batch_size)
        (r_space, e_space), _ = action_space
        action_space_size = r_space.size()[1]

        r_space = r_space.view(batch_size, -1)
        e_space = e_space.view(batch_size, -1)
        log_action_dist = log_action_dist.view(batch_size, -1)
        beam_action_space_size = log_action_dist.size()[1]
        assert (beam_action_space_size % action_space_size == 0)
        k = min(beam_size, beam_action_space_size)
        next_r_list, next_e_list = [], []
        log_action_prob_list = []
        action_offset_list = []
        for i in range(batch_size):
            log_action_dist_b = log_action_dist[i]
            r_space_b = r_space[i]
            e_space_b = e_space[i]
            unique_e_space_b = var_cuda(torch.unique(e_space_b.data.cpu()))
            unique_log_action_dist, unique_idx = unique_max(unique_e_space_b, e_space_b, log_action_dist_b)
            k_prime = min(len(unique_e_space_b), k)
            top_unique_log_action_dist, top_unique_idx2 = torch.topk(unique_log_action_dist, k_prime)
            top_unique_idx = unique_idx[top_unique_idx2]
            top_unique_beam_offset = top_unique_idx / action_space_size
            top_r = r_space_b[top_unique_idx]
            top_e = e_space_b[top_unique_idx]
            next_r_list.append(top_r.unsqueeze(0))
            next_e_list.append(top_e.unsqueeze(0))
            log_action_prob_list.append(top_unique_log_action_dist.unsqueeze(0))
            top_unique_batch_offset = i * last_k
            top_unique_action_offset = top_unique_batch_offset + top_unique_beam_offset
            action_offset_list.append(top_unique_action_offset.unsqueeze(0))
        next_r = ops.pad_and_cat(next_r_list, padding_value=kg.dummy_r).view(-1)
        next_e = ops.pad_and_cat(next_e_list, padding_value=kg.dummy_e).view(-1)
        log_action_prob = ops.pad_and_cat(log_action_prob_list, padding_value=-ops.HUGE_INT)
        action_offset = ops.pad_and_cat(action_offset_list, padding_value=-1)
        return (next_r, next_e), log_action_prob.view(-1), action_offset.view(-1)

    def adjust_search_trace(search_trace, action_offset):
        for i, (r, e) in enumerate(search_trace):
            new_r = r[action_offset]
            new_e = e[action_offset]
            search_trace[i] = (new_r, new_e)

    # Initialization
    r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)  # WHY 最初为啥要空关系
    seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)
    init_action = (r_s, e_s)

    # path encoder
    pn.initialize_path(init_action, kg)
    if kg.args.save_paths_to_csv:
        search_trace = [(r_s, e_s)]

    # Run beam search for num_steps
    # [batch_size*k], k=1
    log_action_prob = zeros_var_cuda(batch_size)
    if return_path_components:
        log_action_probs = []

    action = init_action

    for t in range(num_steps):
        last_r, e = action
        assert (q.size() == e_s.size())
        assert (q.size() == e_t.size())
        assert (e.size()[0] % batch_size == 0)
        assert (q.size()[0] % batch_size == 0)
        k = int(e.size()[0] / batch_size)

        # => [batch_size*k]
        q = ops.tile_along_beam(q.view(batch_size, -1)[:, 0], k)
        e_s = ops.tile_along_beam(e_s.view(batch_size, -1)[:, 0], k)
        e_t = ops.tile_along_beam(e_t.view(batch_size, -1)[:, 0], k)
        obs = [e_s, q, e_t, t == (num_steps - 1), last_r, seen_nodes]
        # one step forward in search
        db_outcomes, _, _ = pn.transit(
            e, obs, kg, use_action_space_bucketing=False,
            merge_aspace_batching_outcome=True)  # TODO:细跟一下里面的get_action_space_in_buckets
        action_space, action_dist = db_outcomes[0]
        # => [batch_size*k, action_space_size]
        log_action_dist = log_action_prob.view(-1, 1) + ops.safe_log(action_dist)
        # [batch_size*k, action_space_size] => [batch_size*new_k]

        if t == num_steps - 1:
            if return_merge_scores is None:
                action, log_action_prob, action_offset = top_k_answer_unique(log_action_dist, action_space)
            else:
                action, log_action_prob, action_offset = top_k_action(log_action_dist, action_space,
                                                                      return_merge_scores)
        else:
            action, log_action_prob, action_offset = top_k_action(log_action_dist, action_space,
                                                                  None)

        if return_path_components:
            ops.rearrange_vector_list(log_action_probs, action_offset)
            log_action_probs.append(log_action_prob)
        pn.update_path(action, kg, offset=action_offset)
        seen_nodes = torch.cat([seen_nodes[action_offset], action[1].unsqueeze(1)], dim=1)
        if kg.args.save_paths_to_csv:
            adjust_search_trace(search_trace, action_offset)
            search_trace.append(action)

    output_beam_size = int(action[0].size()[0] / batch_size)
    # [batch_size*beam_size] => [batch_size, beam_size]
    beam_search_output = dict()
    beam_search_output['pred_e2s'] = action[1].view(batch_size, -1)
    beam_search_output['pred_e2_scores'] = log_action_prob.view(batch_size, -1)

    if return_path_components:
        path_width = 10
        path_components_list = []
        for i in range(batch_size):
            p_c = []
            for k, log_action_prob in enumerate(log_action_probs):
                top_k_edge_labels = []
                for j in range(min(output_beam_size, path_width)):
                    ind = i * output_beam_size + j
                    r = kg.id2relation[int(search_trace[k + 1][0][ind])]
                    e = kg.id2entity[int(search_trace[k + 1][1][ind])]
                    if r.endswith('_inv'):
                        edge_label = '<-{}-{} {}'.format(r[:-4], e, float(log_action_probs[k][ind]))
                    else:
                        edge_label = '-{}->{} {}'.format(r, e, float(log_action_probs[k][ind]))
                    top_k_edge_labels.append(edge_label)
                top_k_action_prob = log_action_prob[:path_width]
                e_name = kg.id2entity[int(search_trace[1][0][i * output_beam_size])] if k == 0 else ''
                p_c.append((e_name, top_k_edge_labels, var_to_numpy(top_k_action_prob)))
            path_components_list.append(p_c)
        beam_search_output['search_traces'] = search_trace
        REAL_ALL_PATHS.append(beam_search_output)

    return beam_search_output


ABS_ALL_PATH = []
ABS_ALL_COMP = []

def beam_search_abs(pn, e_s, q, e_t, kg, num_steps, beam_size, return_path_components=False, return_merge_scores=None):
    """
    Beam search from source.

    :param pn: Policy network.
    :param e_s: (Variable:batch) source entity indices.
    :param q: (Variable:batch) query relation indices.
    :param e_t: (Variable:batch) target entity indices.
    :param kg: Knowledge graph environment.
    :param num_steps: Number of search steps.
    :param beam_size: Beam size used in search.
    :param return_path_components: If set, return all path components at the end of search.
    """
    assert (num_steps >= 1)
    batch_size = len(e_s)
    # print("===>BEAMSEARCH BATCH SIZE:", batch_size)

    def top_k_action(log_action_dist, action_space, return_merge_scores=None):
        """
        Get top k actions.
            - k = beam_size if the beam size is smaller than or equal to the beam action space size
            - k = beam_action_space_size otherwise
        :param log_action_dist: [batch_size*k, action_space_size]
        :param action_space (r_space, e_space):
            r_space: [batch_size*k, action_space_size]
            e_space: [batch_size*k, action_space_size]
        :return:
            (next_r, next_e), action_prob, action_offset: [batch_size*new_k]
        """
        full_size = len(log_action_dist)
        assert (full_size % batch_size == 0)
        last_k = int(full_size / batch_size)

        (r_space, e_space), _ = action_space
        action_space_size = r_space.size()[1]
        # => [batch_size, k'*action_space_size]
        log_action_dist = log_action_dist.view(batch_size, -1)
        beam_action_space_size = log_action_dist.size()[1]
        k = min(beam_size, beam_action_space_size)

        if return_merge_scores is not None:
            if return_merge_scores == 'sum':
                reduce_method = torch.sum
            elif return_merge_scores == 'mean':
                reduce_method = torch.mean
            else:
                reduce_method = None

            all_action_ind = torch.LongTensor([range(beam_action_space_size) for _ in range(len(log_action_dist))]).cuda(device=0)

            all_next_r = ops.batch_lookup(r_space.view(batch_size, -1), all_action_ind)
            all_next_e = ops.batch_lookup(e_space.view(batch_size, -1), all_action_ind)

            real_log_action_prob, real_next_e, real_action_ind, real_next_r = ops.merge_same(log_action_dist, all_next_e, all_next_r, method=reduce_method)

            next_e_list, next_r_list, action_ind_list, log_action_prob_list = [], [], [], []
            for i in range(batch_size):
                k_prime = min(len(real_log_action_prob[i]), k)
                top_log_prob, top_ind = torch.topk(real_log_action_prob[i], k_prime)
                top_next_e, top_next_r, top_ind = real_next_e[i][top_ind], real_next_r[i][top_ind], real_action_ind[i][top_ind]
                next_e_list.append(top_next_e.unsqueeze(0))
                next_r_list.append(top_next_r.unsqueeze(0))
                action_ind_list.append(top_ind.unsqueeze(0))
                log_action_prob_list.append(top_log_prob.unsqueeze(0))

            next_r = ops.pad_and_cat(next_r_list, padding_value=kg.dummy_r).view(-1)
            next_e = ops.pad_and_cat(next_e_list, padding_value=kg.dummy_e).view(-1)
            log_action_prob = ops.pad_and_cat(log_action_prob_list, padding_value=0.0).view(-1)
            action_ind = ops.pad_and_cat(action_ind_list, padding_value=-1).view(-1)
        else:
            log_action_prob, action_ind = torch.topk(log_action_dist, k)
            next_r = ops.batch_lookup(r_space.view(batch_size, -1), action_ind).view(-1)
            next_e = ops.batch_lookup(e_space.view(batch_size, -1), action_ind).view(-1)

            #TODO:CHECK DEBUG 2 DEL
            for e in next_e:
                e = int(e)
                if e not in kg.id2type:
                    print("ERROR e not in kg.id2type:", e)



        # [batch_size, k] => [batch_size*k]
        log_action_prob = log_action_prob.view(-1)
        # *** compute parent offset
        # [batch_size, k]
        action_beam_offset = action_ind / action_space_size
        # [batch_size, 1]
        action_batch_offset = int_var_cuda(torch.arange(batch_size) * last_k).unsqueeze(1)
        # [batch_size, k] => [batch_size*k]
        action_offset = (action_batch_offset + action_beam_offset).view(-1)
        return (next_r, next_e), log_action_prob, action_offset

    def top_k_answer_unique(log_action_dist, action_space):
        """
        Get top k unique entities
            - k = beam_size if the beam size is smaller than or equal to the beam action space size
            - k = beam_action_space_size otherwise
        :param log_action_dist: [batch_size*beam_size, action_space_size] 概率
        :param action_space (r_space, e_space): 实体
            r_space: [batch_size*beam_size, action_space_size]
            e_space: [batch_size*beam_size, action_space_size]
        :return:
            (next_r, next_e), action_prob, action_offset: [batch_size*k]
        """
        full_size = len(log_action_dist)
        assert (full_size % batch_size == 0)
        last_k = int(full_size / batch_size)
        (r_space, e_space), _ = action_space
        action_space_size = r_space.size()[1]

        r_space = r_space.view(batch_size, -1)
        e_space = e_space.view(batch_size, -1)
        log_action_dist = log_action_dist.view(batch_size, -1)
        beam_action_space_size = log_action_dist.size()[1]
        assert (beam_action_space_size % action_space_size == 0)
        k = min(beam_size, beam_action_space_size)
        next_r_list, next_e_list = [], []
        log_action_prob_list = []
        action_offset_list = []
        for i in range(batch_size):
            log_action_dist_b = log_action_dist[i]
            r_space_b = r_space[i]
            e_space_b = e_space[i]
            unique_e_space_b = var_cuda(torch.unique(e_space_b.data.cpu()))
            unique_log_action_dist, unique_idx = unique_max(unique_e_space_b, e_space_b, log_action_dist_b)
            k_prime = min(len(unique_e_space_b), k)
            top_unique_log_action_dist, top_unique_idx2 = torch.topk(unique_log_action_dist, k_prime)
            top_unique_idx = unique_idx[top_unique_idx2]
            top_unique_beam_offset = top_unique_idx / action_space_size
            top_r = r_space_b[top_unique_idx]
            top_e = e_space_b[top_unique_idx]
            next_r_list.append(top_r.unsqueeze(0))
            next_e_list.append(top_e.unsqueeze(0))
            log_action_prob_list.append(top_unique_log_action_dist.unsqueeze(0))
            top_unique_batch_offset = i * last_k
            top_unique_action_offset = top_unique_batch_offset + top_unique_beam_offset
            action_offset_list.append(top_unique_action_offset.unsqueeze(0))
        next_r = ops.pad_and_cat(next_r_list, padding_value=kg.dummy_r).view(-1)
        next_e = ops.pad_and_cat(next_e_list, padding_value=kg.dummy_e).view(-1)

        # TODO:CHECK DEBUG 2 DEL
        for e in next_e:
            e = int(e)
            if e not in kg.id2type:
                print("ERROR IN top_k_answer_unique e not in kg.id2type:", e)


        log_action_prob = ops.pad_and_cat(log_action_prob_list, padding_value=-ops.HUGE_INT)
        action_offset = ops.pad_and_cat(action_offset_list, padding_value=-1)
        return (next_r, next_e), log_action_prob.view(-1), action_offset.view(-1)

    def adjust_search_trace(search_trace, action_offset):
        for i, (r, e) in enumerate(search_trace):
            new_r = r[action_offset]
            new_e = e[action_offset]
            search_trace[i] = (new_r, new_e)

            for e in new_e:
                e = int(e)
                if e not in kg.id2type:
                    print("ERROR IN adjust_search_trace e not in kg.id2type:", e)


    # Initialization
    r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)  # WHY 最初为啥要空关系
    seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)
    init_action = (r_s, e_s)

    # path encoder
    pn.initialize_abs_path(init_action, kg)
    if kg.args.save_paths_to_csv:
        search_trace = [(r_s, e_s)]

    # Run beam search for num_steps
    # [batch_size*k], k=1
    log_action_prob = zeros_var_cuda(batch_size)
    if return_path_components:
        log_action_probs = []

    action = init_action

    for t in range(num_steps):
        last_r, e = action
        assert (q.size() == e_s.size())
        assert (q.size() == e_t.size())
        assert (e.size()[0] % batch_size == 0)
        assert (q.size()[0] % batch_size == 0)
        k = int(e.size()[0] / batch_size)

        # => [batch_size*k]
        q = ops.tile_along_beam(q.view(batch_size, -1)[:, 0], k)
        e_s = ops.tile_along_beam(e_s.view(batch_size, -1)[:, 0], k)
        e_t = ops.tile_along_beam(e_t.view(batch_size, -1)[:, 0], k)
        obs = [e_s, q, e_t, t == (num_steps - 1), last_r, seen_nodes]
        # one step forward in search
        db_outcomes, _, _ = pn.transit_on_abs(
            e, obs, kg, use_action_space_bucketing=False,
            merge_aspace_batching_outcome=True)  # TODO:细跟一下里面的get_action_space_in_buckets
        action_space, action_dist = db_outcomes[0]
        # => [batch_size*k, action_space_size]
        log_action_dist = log_action_prob.view(-1, 1) + ops.safe_log(action_dist)
        # [batch_size*k, action_space_size] => [batch_size*new_k]

        if t == num_steps - 1:
            if return_merge_scores is None:
                action, log_action_prob, action_offset = top_k_answer_unique(log_action_dist, action_space)
            else:
                action, log_action_prob, action_offset = top_k_action(log_action_dist, action_space,
                                                                      return_merge_scores)
        else:
            action, log_action_prob, action_offset = top_k_action(log_action_dist, action_space,
                                                                  None)

        if return_path_components:
            ops.rearrange_vector_list(log_action_probs, action_offset)
            log_action_probs.append(log_action_prob)
        pn.update_path_abs(action, kg, offset=action_offset)
        seen_nodes = torch.cat([seen_nodes[action_offset], action[1].unsqueeze(1)], dim=1)
        if kg.args.save_paths_to_csv:
            adjust_search_trace(search_trace, action_offset)
            search_trace.append(action)

    output_beam_size = int(action[0].size()[0] / batch_size)
    # [batch_size*beam_size] => [batch_size, beam_size]
    beam_search_output = dict()
    beam_search_output['pred_e2s'] = action[1].view(batch_size, -1)
    beam_search_output['pred_e2_scores'] = log_action_prob.view(batch_size, -1)

    # print(search_trace)
    if return_path_components:
        path_width = 10
        path_components_list = []
        for i in range(batch_size):
            p_c = []
            for k, log_action_prob in enumerate(log_action_probs): # k为num_step
                # print (k, len(log_action_prob), log_action_prob)
                top_k_edge_labels = []
                for j in range(min(output_beam_size, path_width)):# 每一个样本多少个path
                    ind = i * output_beam_size + j
                    # print(search_trace[k+1][0])
                    # print(search_trace[k+1][1])
                    # break
                    r = kg.id2relation[int(search_trace[k + 1][0][ind])]
                    _idx = int(search_trace[k + 1][1][ind])
                    e = kg.id2type[_idx]
                    if r.endswith('_inv'):
                        #edge_label = '<=={}-{} {}'.format(r[:-4], e, float(log_action_probs[k][ind]))
                        edge_label = '<=={}-{} '.format(r[:-4], e)
                    else:
                        #edge_label = '-{}==>{} {}'.format(r, e, float(log_action_probs[k][ind]))
                        edge_label = '-{}==>{} '.format(r, e)
                    top_k_edge_labels.append(edge_label)
                top_k_action_prob = log_action_prob[:path_width]
                _idx = int(search_trace[0][1][i * output_beam_size])
                if _idx not in kg.id2type:
                    print ("+++return_path_components, _idx, i, output_beam_size path_width:", _idx, i, output_beam_size, path_width)
                e_name = kg.id2type[_idx] if k == 0 else ''
                p_c.append((e_name, top_k_edge_labels, var_to_numpy(top_k_action_prob)))
            path_components_list.append(p_c)
        beam_search_output['search_traces'] = search_trace
        ABS_ALL_PATH.append(beam_search_output)
        ABS_ALL_COMP.append(path_components_list)
        # print("dumping...")
        # open('use_abs_graph_{}_save_path.json', 'w').write(json.dumps(beam_search_output))
        pickle.dump(beam_search_output,
                    open('use_abs_graph_{}_save_path_idx_{}.pkl'.format(kg.use_abstract_graph, len(ABS_ALL_PATH)-1),
                         'wb')
                    )
        pickle.dump(path_components_list, open('path_components_list_idx_{}.pkl', 'wb'))

    return beam_search_output


SAME_ALL_PATHS = []

def beam_search_same(pn, e_s, q, e_t, kg, num_steps, beam_size, return_path_components=False, return_merge_scores=None, same_start=False):
    """
    Beam search from source.

    :param pn: Policy network.
    :param e_s: (Variable:batch) source entity indices.
    :param q: (Variable:batch) query relation indices.
    :param e_t: (Variable:batch) target entity indices.
    :param kg: Knowledge graph environment.
    :param num_steps: Number of search steps.
    :param beam_size: Beam size used in search.
    :param return_path_components: If set, return all path components at the end of search.
    """
    assert (num_steps >= 1)
    batch_size = len(e_s)

    def top_k_action(log_action_dist, action_space, return_merge_scores=None):
        """
        Get top k actions.
            - k = beam_size if the beam size is smaller than or equal to the beam action space size
            - k = beam_action_space_size otherwise
        :param log_action_dist: [batch_size*k, action_space_size]
        :param action_space (r_space, e_space):
            r_space: [batch_size*k, action_space_size]
            e_space: [batch_size*k, action_space_size]
        :return:
            (next_r, next_e), action_prob, action_offset: [batch_size*new_k]
        """
        full_size = len(log_action_dist)
        assert (full_size % batch_size == 0)
        last_k = int(full_size / batch_size)

        (r_space, e_space), _ = action_space
        action_space_size = r_space.size()[1]
        # => [batch_size, k'*action_space_size]
        log_action_dist = log_action_dist.view(batch_size, -1)
        beam_action_space_size = log_action_dist.size()[1]
        k = min(beam_size, beam_action_space_size)

        if return_merge_scores is not None:
            if return_merge_scores == 'sum':
                reduce_method = torch.sum
            elif return_merge_scores == 'mean':
                reduce_method = torch.mean
            else:
                reduce_method = None

            all_action_ind = torch.LongTensor([range(beam_action_space_size) for _ in range(len(log_action_dist))]).cuda(device=0)

            all_next_r = ops.batch_lookup(r_space.view(batch_size, -1), all_action_ind)
            all_next_e = ops.batch_lookup(e_space.view(batch_size, -1), all_action_ind)

            real_log_action_prob, real_next_e, real_action_ind, real_next_r = ops.merge_same(log_action_dist, all_next_e, all_next_r, method=reduce_method)

            next_e_list, next_r_list, action_ind_list, log_action_prob_list = [], [], [], []
            for i in range(batch_size):
                k_prime = min(len(real_log_action_prob[i]), k)
                top_log_prob, top_ind = torch.topk(real_log_action_prob[i], k_prime)
                top_next_e, top_next_r, top_ind = real_next_e[i][top_ind], real_next_r[i][top_ind], real_action_ind[i][top_ind]
                next_e_list.append(top_next_e.unsqueeze(0))
                next_r_list.append(top_next_r.unsqueeze(0))
                action_ind_list.append(top_ind.unsqueeze(0))
                log_action_prob_list.append(top_log_prob.unsqueeze(0))

            next_r = ops.pad_and_cat(next_r_list, padding_value=kg.dummy_r).view(-1)
            next_e = ops.pad_and_cat(next_e_list, padding_value=kg.dummy_e).view(-1)
            log_action_prob = ops.pad_and_cat(log_action_prob_list, padding_value=0.0).view(-1)
            action_ind = ops.pad_and_cat(action_ind_list, padding_value=-1).view(-1)
        else:
            log_action_prob, action_ind = torch.topk(log_action_dist, k)
            next_r = ops.batch_lookup(r_space.view(batch_size, -1), action_ind).view(-1)
            next_e = ops.batch_lookup(e_space.view(batch_size, -1), action_ind).view(-1)

        # [batch_size, k] => [batch_size*k]
        log_action_prob = log_action_prob.view(-1)
        # *** compute parent offset
        # [batch_size, k]
        action_beam_offset = action_ind / action_space_size
        # [batch_size, 1]
        action_batch_offset = int_var_cuda(torch.arange(batch_size) * last_k).unsqueeze(1)
        # [batch_size, k] => [batch_size*k]
        action_offset = (action_batch_offset + action_beam_offset).view(-1)
        return (next_r, next_e), log_action_prob, action_offset

    def top_k_answer_unique(log_action_dist, action_space):
        """
        Get top k unique entities
            - k = beam_size if the beam size is smaller than or equal to the beam action space size
            - k = beam_action_space_size otherwise
        :param log_action_dist: [batch_size*beam_size, action_space_size] 概率
        :param action_space (r_space, e_space): 实体
            r_space: [batch_size*beam_size, action_space_size]
            e_space: [batch_size*beam_size, action_space_size]
        :return:
            (next_r, next_e), action_prob, action_offset: [batch_size*k]
        """
        full_size = len(log_action_dist)
        assert (full_size % batch_size == 0)
        last_k = int(full_size / batch_size)
        (r_space, e_space), _ = action_space
        action_space_size = r_space.size()[1]

        r_space = r_space.view(batch_size, -1)
        e_space = e_space.view(batch_size, -1)
        log_action_dist = log_action_dist.view(batch_size, -1)
        beam_action_space_size = log_action_dist.size()[1]
        assert (beam_action_space_size % action_space_size == 0)
        k = min(beam_size, beam_action_space_size)
        next_r_list, next_e_list = [], []
        log_action_prob_list = []
        action_offset_list = []
        for i in range(batch_size):
            log_action_dist_b = log_action_dist[i]
            r_space_b = r_space[i]
            e_space_b = e_space[i]
            unique_e_space_b = var_cuda(torch.unique(e_space_b.data.cpu()))
            unique_log_action_dist, unique_idx = unique_max(unique_e_space_b, e_space_b, log_action_dist_b)
            k_prime = min(len(unique_e_space_b), k)
            top_unique_log_action_dist, top_unique_idx2 = torch.topk(unique_log_action_dist, k_prime)
            top_unique_idx = unique_idx[top_unique_idx2]
            top_unique_beam_offset = top_unique_idx / action_space_size
            top_r = r_space_b[top_unique_idx]
            top_e = e_space_b[top_unique_idx]
            next_r_list.append(top_r.unsqueeze(0))
            next_e_list.append(top_e.unsqueeze(0))
            log_action_prob_list.append(top_unique_log_action_dist.unsqueeze(0))
            top_unique_batch_offset = i * last_k
            top_unique_action_offset = top_unique_batch_offset + top_unique_beam_offset
            action_offset_list.append(top_unique_action_offset.unsqueeze(0))
        next_r = ops.pad_and_cat(next_r_list, padding_value=kg.dummy_r).view(-1)
        next_e = ops.pad_and_cat(next_e_list, padding_value=kg.dummy_e).view(-1)
        log_action_prob = ops.pad_and_cat(log_action_prob_list, padding_value=-ops.HUGE_INT)
        action_offset = ops.pad_and_cat(action_offset_list, padding_value=-1)
        return (next_r, next_e), log_action_prob.view(-1), action_offset.view(-1)

    def adjust_search_trace(search_trace, action_offset):
        for i, (r, e) in enumerate(search_trace):
            new_r = r[action_offset]
            new_e = e[action_offset]
            search_trace[i] = (new_r, new_e)

    # Initialization
    r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)  # WHY 最初为啥要空关系
    seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1) #TODO
    init_action = (r_s, e_s)
    e_s_abs = var_cuda(torch.LongTensor([kg.get_typeid(_) for _ in e_s]), requires_grad=False)
    e_t_abs = var_cuda(torch.LongTensor([kg.get_typeid(_) for _ in e_t]), requires_grad=False)
    # print("()()()es_size() e_s_abs.size():", e_s.size(), e_s_abs.size())
    seen_nodes_abs = int_fill_var_cuda(e_s_abs.size(), kg.dummy_e).unsqueeze(1)
    init_action_abs = (r_s, e_s_abs)
    init_action = (r_s, e_s)

    # path encoder
    # pn.initialize_path(init_action, kg)
    pn.initialize_abs_path(init_action, init_action_abs, kg, same_start)
    if kg.args.save_paths_to_csv:
        search_trace = [(r_s, e_s)]

    # Run beam search for num_steps
    # [batch_size*k], k=1
    log_action_prob = zeros_var_cuda(batch_size)
    if return_path_components:
        log_action_probs = []

    action = init_action
    action_abs = init_action_abs

    for t in range(num_steps):
        last_r, e = action
        last_r, e_abs = action_abs
        assert (q.size() == e_s.size())
        assert (q.size() == e_t.size())
        assert (e.size()[0] % batch_size == 0)
        assert (q.size()[0] % batch_size == 0)
        k = int(e.size()[0] / batch_size)

        # => [batch_size*k]
        q = ops.tile_along_beam(q.view(batch_size, -1)[:, 0], k)
        e_s = ops.tile_along_beam(e_s.view(batch_size, -1)[:, 0], k)
        e_s_abs = ops.tile_along_beam(e_abs.view(batch_size, -1)[:, 0], k)
        e_t = ops.tile_along_beam(e_t.view(batch_size, -1)[:, 0], k)
        e_t_abs = ops.tile_along_beam(e_t_abs.view(batch_size, -1)[:, 0], k)
        obs = [e_s, q, e_t, t == (num_steps - 1), last_r, seen_nodes]
        obs_abs = [e_s_abs, q, e_t_abs, t == (num_steps - 1), last_r, seen_nodes_abs]
        # one step forward in search
        # db_outcomes, _, _ = pn.transit_same(
        #     e, obs, kg, use_action_space_bucketing=False,
        #     merge_aspace_batching_outcome=True)  # TODO:细跟一下里面的get_action_space_in_buckets
        db_outcomes, _, _ = pn.transit_same(
            e, obs, e_abs, obs_abs, kg, use_action_space_bucketing=False,
            merge_aspace_batching_outcome=True)  # TODO:细跟一下里面的get_action_space_in_buckets
        action_space, action_dist = db_outcomes[0]
        # => [batch_size*k, action_space_size]
        log_action_dist = log_action_prob.view(-1, 1) + ops.safe_log(action_dist)
        # [batch_size*k, action_space_size] => [batch_size*new_k]

        if t == num_steps - 1:
            if return_merge_scores is None:
                action, log_action_prob, action_offset = top_k_answer_unique(log_action_dist, action_space)
            else:
                action, log_action_prob, action_offset = top_k_action(log_action_dist, action_space,
                                                                      return_merge_scores)
        else:
            action, log_action_prob, action_offset = top_k_action(log_action_dist, action_space,
                                                                  None)

        #(next_r, next_e)
        action_abs = ( action[0], var_cuda(torch.LongTensor([kg.get_typeid(_) for _ in action[1]]), requires_grad=False) )

        if return_path_components:
            ops.rearrange_vector_list(log_action_probs, action_offset)
            log_action_probs.append(log_action_prob)
        pn.update_path_abs(action_abs, kg, offset=action_offset)
        seen_nodes = torch.cat([seen_nodes[action_offset], action[1].unsqueeze(1)], dim=1)
        seen_nodes_abs = torch.cat([seen_nodes_abs[action_offset], action_abs[1].unsqueeze(1)], dim=1)
        if kg.args.save_paths_to_csv:
            adjust_search_trace(search_trace, action_offset)
            search_trace.append(action)

    output_beam_size = int(action[0].size()[0] / batch_size)
    # [batch_size*beam_size] => [batch_size, beam_size]
    beam_search_output = dict()
    beam_search_output['pred_e2s'] = action[1].view(batch_size, -1)
    beam_search_output['pred_e2_scores'] = log_action_prob.view(batch_size, -1)

    if return_path_components:
        path_width = 10
        path_components_list = []
        for i in range(batch_size):
            p_c = []
            for k, log_action_prob in enumerate(log_action_probs):
                top_k_edge_labels = []
                for j in range(min(output_beam_size, path_width)):
                    ind = i * output_beam_size + j
                    r = kg.id2relation[int(search_trace[k + 1][0][ind])]
                    e = kg.id2entity[int(search_trace[k + 1][1][ind])]
                    if r.endswith('_inv'):
                        edge_label = '<-{}-{} {}'.format(r[:-4], e, float(log_action_probs[k][ind]))
                    else:
                        edge_label = '-{}->{} {}'.format(r, e, float(log_action_probs[k][ind]))
                    top_k_edge_labels.append(edge_label)
                top_k_action_prob = log_action_prob[:path_width]
                e_name = kg.id2entity[int(search_trace[1][0][i * output_beam_size])] if k == 0 else ''
                p_c.append((e_name, top_k_edge_labels, var_to_numpy(top_k_action_prob)))
            path_components_list.append(p_c)
        beam_search_output['search_traces'] = search_trace
        SAME_ALL_PATHS.append(beam_search_output)

    return beam_search_output
