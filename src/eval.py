"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Compute Evaluation Metrics.
 Code adapted from https://github.com/TimDettmers/ConvE/blob/master/evaluation.py
"""
import copy
import numpy as np
import pickle

import torch

from src.parse_args import args
from src.data_utils import NO_OP_ENTITY_ID, DUMMY_ENTITY_ID


def hits_and_ranks(examples, scores, all_answers, verbose=False):
    """
    Compute ranking based metrics.
    """
    assert (len(examples) == scores.shape[0])
    # mask false negatives in the predictions
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    for i, example in enumerate(examples):
        e1, e2, r = example
        e2_multi = dummy_mask + list(all_answers[e1][r])
        # save the relevant prediction
        target_score = float(scores[i, e2])
        # mask all false negatives
        scores[i, e2_multi] = 0
        # write back the save prediction
        scores[i, e2] = target_score

    # sort and rank
    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), args.beam_size))
    top_k_targets = top_k_targets.cpu().numpy()

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    mrr = 0
    for i, example in enumerate(examples):
        e1, e2, r = example
        pos = np.where(top_k_targets[i] == e2)[0]
        if len(pos) > 0:
            pos = pos[0]
            if pos < 10:
                hits_at_10 += 1
                if pos < 5:
                    hits_at_5 += 1
                    if pos < 3:
                        hits_at_3 += 1
                        if pos < 1:
                            hits_at_1 += 1
            mrr += 1.0 / (pos + 1)

    hits_at_1 = float(hits_at_1) / len(examples)
    hits_at_3 = float(hits_at_3) / len(examples)
    hits_at_5 = float(hits_at_5) / len(examples)
    hits_at_10 = float(hits_at_10) / len(examples)
    mrr = float(mrr) / len(examples)

    if verbose:
        print('Hits@1 = {}'.format(hits_at_1))
        print('Hits@3 = {}'.format(hits_at_3))
        print('Hits@5 = {}'.format(hits_at_5))
        print('Hits@10 = {}'.format(hits_at_10))
        print('MRR = {}'.format(mrr))

    return hits_at_1, hits_at_3, hits_at_5, hits_at_10, mrr


def hits_and_ranks_merge(examples, scores, all_answers, scores_real, lmd, e2t, verbose=False):
    """
        Compute ranking based metrics.
        """
    assert (len(examples) == scores.shape[0])
    # mask false negatives in the predictions
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    for i, example in enumerate(examples):
        e1, e2, r = example
        e2_multi = dummy_mask + list(all_answers[e1][r])
        # save the relevant prediction
        target_score_1 = float(scores[i, e2])
        target_score_2 = float(scores_real[i, e2])
        if target_score_1 != 0 and target_score_2 != 0:
            target_score = lmd * target_score_1 + (1 - lmd) * target_score_2
        else:
            target_score = (1 - lmd) * target_score_2  # TODO:CHECK
            # target_score = (1 - lmd) *  target_score_2 #TODO:CHECK
        # mask all false negatives
        scores[i, e2_multi] = 0
        # write back the save prediction
        scores[i, e2] = target_score

    # sort and rank
    # print("+++++++>> min(scores.size(1), args.beam_size:",   scores.size(1), args.beam_size, scores)
    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), args.beam_size))
    top_k_targets = top_k_targets.cpu().numpy()

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    mrr = 0
    for i, example in enumerate(examples):
        e1, e2, r = example
        pos = np.where(top_k_targets[i] == e2)[0]
        if len(pos) > 0:
            pos = pos[0]
            if pos < 10:
                hits_at_10 += 1
                if pos < 5:
                    hits_at_5 += 1
                    if pos < 3:
                        hits_at_3 += 1
                        if pos < 1:
                            hits_at_1 += 1
            mrr += 1.0 / (pos + 1)

    hits_at_1 = float(hits_at_1) / len(examples)
    hits_at_3 = float(hits_at_3) / len(examples)
    hits_at_5 = float(hits_at_5) / len(examples)
    hits_at_10 = float(hits_at_10) / len(examples)
    mrr = float(mrr) / len(examples)

    if verbose:
        print('Hits@1 = {}'.format(hits_at_1))
        print('Hits@3 = {}'.format(hits_at_3))
        print('Hits@5 = {}'.format(hits_at_5))
        print('Hits@10 = {}'.format(hits_at_10))
        print('MRR = {}'.format(mrr))

    return hits_at_1, hits_at_3, hits_at_5, hits_at_10, mrr


# def hits_and_ranks_merge_inner(examples, scores, all_answers, scores_real, lmd, e2t, verbose=False):
#     """
#         Compute ranking based metrics.
#         """
#     assert (len(examples) == scores.shape[0])
#     # mask false negatives in the predictions
#     dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
#     for i, example in enumerate(examples):
#         e1, e2, r = example
#         e2_multi = dummy_mask + list(all_answers[e1][r])
#         # save the relevant prediction
#         # target_score_1 = float(scores[i, e2])
#         target_score_2 = float(scores_real[i, e2])
#         # if target_score_1 != 0 and target_score_2 != 0:
#         #     target_score = lmd * target_score_1 + (1 - lmd) * target_score_2
#         # else:
#         #     target_score = (1 - lmd) * target_score_2 #TODO:CHECK
#         # target_score = (1 - lmd) *  target_score_2 #TODO:CHECK
#         # mask all false negatives
#         scores[i, e2_multi] = 0
#         # write back the save prediction
#         scores[i, e2] = target_score_2
#
#     # sort and rank
#     # print("+++++++>> min(scores.size(1), args.beam_size:",   scores.size(1), args.beam_size, scores)
#     top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), args.beam_size))
#     top_k_targets = top_k_targets.cpu().numpy()
#
#     ans = []
#     for i, example in enumerate(top_k_targets):
#         type_list = []
#         target_type_list = []
#
#         for p, e in zip(top_k_scores[i], top_k_targets[i]):
#             if p != 0.0:
#                 p += scores[i, e]
#             if len(target_type_list) == 0 or type_list[-1] != e2t[e]:
#                 type_list.append(e2t[e])
#                 target_type_list.append([(e, p), ])
#             else:
#                 target_type_list[-1].append((e, p))
#         real_target_list = []
#         for chunk in target_type_list:
#             chunk = sorted(chunk, key=lambda d: d[1], reverse=True)
#             real_target_list += [_[0] for _ in chunk]
#         assert len(real_target_list) == len(top_k_targets[i])
#         ans.append(real_target_list)
#
#     ans = np.asarray(ans, dtype=np.int32)
#     top_k_targets = ans
#
#     hits_at_1 = 0
#     hits_at_3 = 0
#     hits_at_5 = 0
#     hits_at_10 = 0
#     mrr = 0
#     for i, example in enumerate(examples):
#         e1, e2, r = example
#         pos = np.where(top_k_targets[i] == e2)[0]
#         if len(pos) > 0:
#             pos = pos[0]
#             if pos < 10:
#                 hits_at_10 += 1
#                 if pos < 5:
#                     hits_at_5 += 1
#                     if pos < 3:
#                         hits_at_3 += 1
#                         if pos < 1:
#                             hits_at_1 += 1
#             mrr += 1.0 / (pos + 1)
#
#     hits_at_1 = float(hits_at_1) / len(examples)
#     hits_at_3 = float(hits_at_3) / len(examples)
#     hits_at_5 = float(hits_at_5) / len(examples)
#     hits_at_10 = float(hits_at_10) / len(examples)
#     mrr = float(mrr) / len(examples)
#
#     if verbose:
#         print('Hits@1 = {}'.format(hits_at_1))
#         print('Hits@3 = {}'.format(hits_at_3))
#         print('Hits@5 = {}'.format(hits_at_5))
#         print('Hits@10 = {}'.format(hits_at_10))
#         print('MRR = {}'.format(mrr))
#
#     return hits_at_1, hits_at_3, hits_at_5, hits_at_10, mrr


def hits_and_ranks_merge_inner(examples, baseline_scores, all_answers, scores_real, lmd, e2t, verbose=False):
    """
        Compute ranking based metrics.
        """
    assert (len(examples) == baseline_scores.shape[0])
    assert (len(examples) == scores_real.shape[0])
    # mask false negatives in the predictions
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    scores = copy.deepcopy(scores_real)
    for i, example in enumerate(examples):
        e1, e2, r = example
        e2_multi = dummy_mask + list(all_answers[e1][r])
        # save the relevant prediction
        # target_score_1 = float(scores[i, e2])
        target_score_2 = float(scores_real[i, e2])
        # if target_score_1 != 0 and target_score_2 != 0:
        #     target_score = lmd * target_score_1 + (1 - lmd) * target_score_2
        # else:
        #     target_score = (1 - lmd) * target_score_2 #TODO:CHECK
        # target_score = (1 - lmd) *  target_score_2 #TODO:CHECK
        # mask all false negatives
        scores[i, e2_multi] = 0
        # write back the save prediction
        scores[i, e2] = target_score_2

    # sort and rank
    # print("+++++++>> min(scores.size(1), args.beam_size:",   scores.size(1), args.beam_size, scores)
    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), args.beam_size))
    top_k_targets = top_k_targets.cpu().numpy()

    ans = []
    for i, example in enumerate(top_k_targets):
        # type_list = []
        target_type_list = []
        type2idx = {}

        for p, e in zip(top_k_scores[i], top_k_targets[i]):
            if p != 0.0:
                p += baseline_scores[i, e]
            e_type = e2t[e]
            if e_type not in type2idx:
                sz = len(type2idx)
                type2idx[e_type] = sz
                target_type_list.append([])
                # type_list.append(e_type)

            target_type_list[type2idx[e_type]].append((e, p))

            # if len(target_type_list) == 0 or type_list[-1] != e2t[e]:
            #     type_list.append(e2t[e])
            #     target_type_list.append([(e, p), ])
            # else:
            #     target_type_list[-1].append((e, p))
        real_target_list = []
        for chunk in target_type_list:
            chunk = sorted(chunk, key=lambda d: d[1], reverse=True)
            real_target_list += [_[0] for _ in chunk]
        assert len(real_target_list) == len(top_k_targets[i])
        ans.append(real_target_list)

    ans = np.asarray(ans, dtype=np.int32)
    top_k_targets = ans

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    mrr = 0
    for i, example in enumerate(examples):
        e1, e2, r = example
        pos = np.where(top_k_targets[i] == e2)[0]
        if len(pos) > 0:
            pos = pos[0]
            if pos < 10:
                hits_at_10 += 1
                if pos < 5:
                    hits_at_5 += 1
                    if pos < 3:
                        hits_at_3 += 1
                        if pos < 1:
                            hits_at_1 += 1
            mrr += 1.0 / (pos + 1)

    hits_at_1 = float(hits_at_1) / len(examples)
    hits_at_3 = float(hits_at_3) / len(examples)
    hits_at_5 = float(hits_at_5) / len(examples)
    hits_at_10 = float(hits_at_10) / len(examples)
    mrr = float(mrr) / len(examples)

    if verbose:
        print('Hits@1 = {}'.format(hits_at_1))
        print('Hits@3 = {}'.format(hits_at_3))
        print('Hits@5 = {}'.format(hits_at_5))
        print('Hits@10 = {}'.format(hits_at_10))
        print('MRR = {}'.format(mrr))

    return hits_at_1, hits_at_3, hits_at_5, hits_at_10, mrr


def hits_at_k(examples, scores, all_answers, verbose=False):
    """
    Hits at k metrics.
    :param examples: List of triples and labels (+/-).
    :param pred_targets:
    :param scores:
    :param all_answers:
    :param verbose:
    """
    assert (len(examples) == scores.shape[0])
    # mask false negatives in the predictions
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    for i, example in enumerate(examples):
        e1, e2, r = example
        e2_multi = list(all_answers[e1][r]) + dummy_mask
        # save the relevant prediction
        target_score = scores[i, e2]
        # mask all false negatives
        scores[i][e2_multi] = 0
        scores[i][dummy_mask] = 0
        # write back the save prediction
        scores[i][e2] = target_score

    # sort and rank
    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), args.beam_size))
    top_k_targets = top_k_targets.cpu().numpy()

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    for i, example in enumerate(examples):
        e1, e2, r = example
        pos = np.where(top_k_targets[i] == e2)[0]
        if pos:
            pos = pos[0]
            if pos < 10:
                hits_at_10 += 1
                if pos < 5:
                    hits_at_5 += 1
                    if pos < 3:
                        hits_at_3 += 1
                        if pos < 1:
                            hits_at_1 += 1

    hits_at_1 = float(hits_at_1) / len(examples)
    hits_at_3 = float(hits_at_3) / len(examples)
    hits_at_5 = float(hits_at_5) / len(examples)
    hits_at_10 = float(hits_at_10) / len(examples)

    if verbose:
        print('Hits@1 = {}'.format(hits_at_1))
        print('Hits@3 = {}'.format(hits_at_3))
        print('Hits@5 = {}'.format(hits_at_5))
        print('Hits@10 = {}'.format(hits_at_10))

    return hits_at_1, hits_at_3, hits_at_5, hits_at_10


def hits_and_ranks_by_seen_queries(examples, scores, all_answers, seen_queries, verbose=False):
    seen_exps, unseen_exps = [], []
    seen_ids, unseen_ids = [], []
    for i, example in enumerate(examples):
        e1, e2, r = example
        if (e1, r) in seen_queries:
            seen_exps.append(example)
            seen_ids.append(i)
        else:
            unseen_exps.append(example)
            unseen_ids.append(i)

    _, _, _, _, seen_mrr = hits_and_ranks(seen_exps, scores[seen_ids], all_answers, verbose=False)
    _, _, _, _, unseen_mrr = hits_and_ranks(unseen_exps, scores[unseen_ids], all_answers, verbose=False)
    if verbose:
        print('MRR on seen queries: {}'.format(seen_mrr))
        print('MRR on unseen queries: {}'.format(unseen_mrr))
    return seen_mrr, unseen_mrr


def hits_and_ranks_by_relation_type(examples, scores, all_answers, relation_by_types, verbose=False):
    to_M_rels, to_1_rels = relation_by_types
    to_M_exps, to_1_exps = [], []
    to_M_ids, to_1_ids = [], []
    for i, example in enumerate(examples):
        e1, e2, r = example
        if r in to_M_rels:
            to_M_exps.append(example)
            to_M_ids.append(i)
        else:
            to_1_exps.append(example)
            to_1_ids.append(i)

    _, _, _, _, to_m_mrr = hits_and_ranks(to_M_exps, scores[to_M_ids], all_answers, verbose=False)
    _, _, _, _, to_1_mrr = hits_and_ranks(to_1_exps, scores[to_1_ids], all_answers, verbose=False)
    if verbose:
        print('MRR on to-M relations: {}'.format(to_m_mrr))
        print('MRR on to-1 relations: {}'.format(to_1_mrr))
    return to_m_mrr, to_1_mrr


def link_MAP(examples, scores, labels, all_answers, verbose=False):
    """
    Per-query mean average precision.
    """
    assert (len(examples) == len(scores))
    queries = {}
    for i, example in enumerate(examples):
        e1, e2, r = example
        if not e1 in queries:
            queries[e1] = []
        queries[e1].append((examples[i], labels[i], scores[i][e2]))

    aps = []
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]

    for e1 in queries:
        ranked_examples = sorted(queries[e1], key=lambda x: x[2], reverse=True)
        acc_precision, offset, num_pos = 0, 0, 0
        for i in range(len(ranked_examples)):
            triple, label, score = ranked_examples[i]
            _, r, e2 = triple
            if label == '+':
                num_pos += 1
                acc_precision += float(num_pos) / (i + 1 - offset)
            else:
                answer_set = {}
                if e1 in all_answers and r in all_answers[e1]:
                    answer_set = all_answers[e1][r]
                if e2 in answer_set or e2 in dummy_mask:
                    print('False negative found: {}'.format(triple))
                    offset += 1
        if num_pos > 0:
            ap = acc_precision / num_pos
            aps.append(ap)
    map = np.mean(aps)
    if verbose:
        print('MAP = {}'.format(map))
    return map


def export_error_cases(examples, scores, all_answers, output_path):
    """
    Export indices of examples to which the top-1 prediction is incorrect.
    """
    assert (len(examples) == scores.shape[0])
    # mask false negatives in the predictions
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    for i, example in enumerate(examples):
        e1, e2, r = example
        e2_multi = dummy_mask + list(all_answers[e1][r])
        # save the relevant prediction
        target_score = float(scores[i, e2])
        # mask all false negatives
        scores[i, e2_multi] = 0
        # write back the save prediction
        scores[i, e2] = target_score

    # sort and rank
    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), args.beam_size))
    top_k_targets = top_k_targets.cpu().numpy()

    top_1_errors, top_10_errors = [], []
    for i, example in enumerate(examples):
        e1, e2, r = example
        pos = np.where(top_k_targets[i] == e2)[0]
        if len(pos) <= 0 or pos[0] > 0:
            top_1_errors.append(i)
        if len(pos) <= 0 or pos[0] > 9:
            top_10_errors.append(i)
    with open(output_path, 'wb') as o_f:
        pickle.dump([top_1_errors, top_10_errors], o_f)

    print('{}/{} top-1 error cases written to {}'.format(len(top_1_errors), len(examples), output_path))
    print('{}/{} top-10 error cases written to {}'.format(len(top_10_errors), len(examples), output_path))
