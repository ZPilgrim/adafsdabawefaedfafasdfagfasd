#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Experiment Portal.
"""
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ''
# print("USE GPU 1,2")
import copy
import itertools
import numpy as np
import numpy as np
import os, sys
import random
import collections
import numpy as np
import torch

from src.parse_args import parser
from src.parse_args import args
import src.data_utils as data_utils
import src.eval
from src.hyperparameter_range import hp_range
from src.knowledge_graph import KnowledgeGraph
from src.emb.fact_network import ComplEx, ConvE, DistMult
from src.emb.fact_network import get_conve_kg_state_dict, get_complex_kg_state_dict, get_distmult_kg_state_dict
from src.emb.emb import EmbeddingBasedMethod
from src.rl.graph_search.pn import GraphSearchPolicy
from src.rl.graph_search.pg import PolicyGradient
from src.rl.graph_search.rs_pg import RewardShapingPolicyGradient
from src.utils.ops import flatten, zeros_var_cuda
import pickle

torch.cuda.set_device(args.gpu)
# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# print('======>args.use_action_space_bucketing = False')
# args.use_action_space_bucketing = False


def process_data():
    data_dir = args.data_dir
    raw_kb_path = os.path.join(data_dir, 'raw.kb')
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(data_dir, 'dev.triples')
    test_path = os.path.join(data_dir, 'test.triples')
    data_utils.prepare_kb_envrioment(raw_kb_path, train_path, dev_path, test_path, args.test, args.bandwidth,
                                     args.add_reverse_relations)


def initialize_model_directory(args, random_seed=None):
    # add model parameter info to model directory
    model_root_dir = args.model_root_dir
    dataset = os.path.basename(os.path.normpath(args.data_dir))

    reverse_edge_tag = '-RV' if args.add_reversed_training_edges else ''
    entire_graph_tag = '-EG' if args.train_entire_graph else ''
    if args.xavier_initialization:
        initialization_tag = '-xavier'
    elif args.uniform_entity_initialization:
        initialization_tag = '-uniform'
    else:
        initialization_tag = ''

    # Hyperparameter signature
    if args.model in ['rule']:
        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.baseline,
            args.entity_dim,
            args.relation_dim,
            args.history_num_layers,
            args.learning_rate,
            args.emb_dropout_rate,
            args.ff_dropout_rate,
            args.action_dropout_rate,
            args.bandwidth,
            args.beta
        )
    elif args.model.startswith('point'):
        if args.baseline == 'avg_reward':
            print('* Policy Gradient Baseline: average reward')
        elif args.baseline == 'avg_reward_normalized':
            print('* Policy Gradient Baseline: average reward baseline plus normalization')
        else:
            print('* Policy Gradient Baseline: None')
        if args.action_dropout_anneal_interval < 1000:
            hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                args.baseline,
                args.entity_dim,
                args.relation_dim,
                args.history_num_layers,
                args.learning_rate,
                args.emb_dropout_rate,
                args.ff_dropout_rate,
                args.action_dropout_rate,
                args.action_dropout_anneal_factor,
                args.action_dropout_anneal_interval,
                args.bandwidth,
                args.beta
            )
            if args.mu != 1.0:
                hyperparam_sig += '-{}'.format(args.mu)
        else:
            hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                args.baseline,
                args.entity_dim,
                args.relation_dim,
                args.history_num_layers,
                args.learning_rate,
                args.emb_dropout_rate,
                args.ff_dropout_rate,
                args.action_dropout_rate,
                args.bandwidth,
                args.beta
            )
        if args.reward_shaping_threshold > 0:
            hyperparam_sig += '-{}'.format(args.reward_shaping_threshold)
    elif args.model == 'distmult':
        hyperparam_sig = '{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.emb_dropout_rate,
            args.label_smoothing_epsilon
        )
    elif args.model == 'complex':
        hyperparam_sig = '{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.emb_dropout_rate,
            args.label_smoothing_epsilon
        )
    elif args.model in ['conve', 'hypere', 'triplee']:
        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.num_out_channels,
            args.kernel_size,
            args.emb_dropout_rate,
            args.hidden_dropout_rate,
            args.feat_dropout_rate,
            args.label_smoothing_epsilon
        )
    else:
        raise NotImplementedError

    model_sub_dir = '{}-{}{}{}{}-{}'.format(
        dataset,
        args.model,
        reverse_edge_tag,
        entire_graph_tag,
        initialization_tag,
        hyperparam_sig
    )
    if args.model == 'set':
        model_sub_dir += '-{}'.format(args.beam_size)
        model_sub_dir += '-{}'.format(args.num_paths_per_entity)
    if args.relation_only:
        model_sub_dir += '-ro'
    elif args.relation_only_in_path:
        model_sub_dir += '-rpo'
    elif args.type_only:
        model_sub_dir += '-to'

    if args.test:
        model_sub_dir += '-test'

    if random_seed:
        model_sub_dir += '.{}'.format(random_seed)

    model_dir = os.path.join(model_root_dir, model_sub_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print('Model directory created: {}'.format(model_dir))
    else:
        print('Model directory exists: {}'.format(model_dir))

    args.model_dir = model_dir


def construct_model(args):
    """
    Construct NN graph.
    """
    kg = KnowledgeGraph(args)
    if args.model.endswith('.gc'):
        kg.load_fuzzy_facts()

    if args.model in ['point', 'point.gc']:
        pn = GraphSearchPolicy(kg, args)
        lf = PolicyGradient(args, kg, pn)
    elif args.model.startswith('point.rs'):
        pn = GraphSearchPolicy(kg, args)
        fn_model = args.model.split('.')[2]
        fn_args = copy.deepcopy(args)
        print("Force fn_kg use_abstract_graph to False:")
        fn_args.use_abstract_graph = False
        fn_args.model = fn_model
        fn_args.relation_only = False
        if fn_model == 'complex':
            fn = ComplEx(fn_args)
            fn_kg = KnowledgeGraph(fn_args)
        elif fn_model == 'distmult':
            fn = DistMult(fn_args)
            fn_kg = KnowledgeGraph(fn_args)
        elif fn_model == 'conve':
            fn = ConvE(fn_args, kg.num_entities)
            fn_kg = KnowledgeGraph(fn_args)
        lf = RewardShapingPolicyGradient(args, kg, pn, fn_kg, fn)
    elif args.model == 'complex':
        fn = ComplEx(args)
        lf = EmbeddingBasedMethod(args, kg, fn)
    elif args.model == 'distmult':
        fn = DistMult(args)
        lf = EmbeddingBasedMethod(args, kg, fn)
    elif args.model == 'conve':
        fn = ConvE(args, kg.num_entities)
        lf = EmbeddingBasedMethod(args, kg, fn)
    else:
        raise NotImplementedError
    return lf


def train(lf):
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    train_data = data_utils.load_triples(
        train_path, entity_index_path, relation_index_path, group_examples_by_query=args.group_examples_by_query,
        add_reverse_relations=args.add_reversed_training_edges)
    if 'NELL' in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities)
    if args.checkpoint_path is not None:
        lf.load_checkpoint(args.checkpoint_path)
    if args.use_abstract_graph:
        lf.run_train_with_abstract(train_data, dev_data)
    else:
        lf.run_train(train_data, dev_data)


def inference(lf):
    lf.batch_size = args.dev_batch_size
    lf.eval()
    # TODO:CHECK load model
    if args.model == 'hypere':
        conve_kg_state_dict = get_conve_kg_state_dict(torch.load(args.conve_state_dict_path))
        lf.kg.load_state_dict(conve_kg_state_dict)
        secondary_kg_state_dict = get_complex_kg_state_dict(torch.load(args.complex_state_dict_path))
        lf.secondary_kg.load_state_dict(secondary_kg_state_dict)
    elif args.model == 'triplee':
        conve_kg_state_dict = get_conve_kg_state_dict(torch.load(args.conve_state_dict_path))
        lf.kg.load_state_dict(conve_kg_state_dict)
        complex_kg_state_dict = get_complex_kg_state_dict(torch.load(args.complex_state_dict_path))
        lf.secondary_kg.load_state_dict(complex_kg_state_dict)
        distmult_kg_state_dict = get_distmult_kg_state_dict(torch.load(args.distmult_state_dict_path))
        lf.tertiary_kg.load_state_dict(distmult_kg_state_dict)
    else:
        lf.load_checkpoint(get_checkpoint_path(args))
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')

    with open(os.path.join(args.data_dir, 'entity2typeid.pkl'), 'rb') as f:
        print("loading entity2typeid.pkl")
        entity2typeid = pickle.load(f)
    if 'NELL' in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
        adj_list_abs_path = os.path.join(args.data_dir, 'adj_list_abs.pkl')
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
        seen_entities_abs = set([entity2typeid[_] for _ in entity2typeid])
    else:
        seen_entities = set()
        seen_entities_abs = set()

    eval_metrics = {
        'dev': {},
        'test': {}
    }

    if args.compute_map:
        relation_sets = [
            'concept:athletehomestadium',
            'concept:athleteplaysforteam',
            'concept:athleteplaysinleague',
            'concept:athleteplayssport',
            'concept:organizationheadquarteredincity',
            'concept:organizationhiredperson',
            'concept:personborninlocation',
            'concept:teamplayssport',
            'concept:worksfor'
        ]
        mps = []
        for r in relation_sets:
            print('* relation: {}'.format(r))
            test_path = os.path.join(args.data_dir, 'tasks', r, 'test.pairs')
            test_data, labels = data_utils.load_triples_with_label(
                test_path, r, entity_index_path, relation_index_path, seen_entities=seen_entities)
            pred_scores = lf.forward(test_data, verbose=False)
            mp = src.eval.link_MAP(test_data, pred_scores, labels, lf.kg.all_objects, verbose=True)
            mps.append(mp)

        map_ = np.mean(mps)
        print('Overall MAP = {}'.format(map_))
        eval_metrics['test']['avg_map'] = map
    elif args.eval_by_relation_type:
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path,
                                           seen_entities=seen_entities)
        pred_scores = lf.forward(dev_data, verbose=False)
        to_m_rels, to_1_rels, _ = data_utils.get_relations_by_type(args.data_dir, relation_index_path)
        relation_by_types = (to_m_rels, to_1_rels)
        print('Dev set evaluation by relation type (partial graph)')
        src.eval.hits_and_ranks_by_relation_type(
            dev_data, pred_scores, lf.kg.dev_objects, relation_by_types, verbose=True)
        print('Dev set evaluation by relation type (full graph)')
        src.eval.hits_and_ranks_by_relation_type(
            dev_data, pred_scores, lf.kg.all_objects, relation_by_types, verbose=True)
    elif args.eval_by_seen_queries:
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path,
                                           seen_entities=seen_entities)
        pred_scores = lf.forward(dev_data, verbose=False)
        seen_queries = data_utils.get_seen_queries(args.data_dir, entity_index_path, relation_index_path)

        print('Dev set evaluation by seen queries (partial graph)')
        src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.dev_objects, seen_queries, verbose=True)
        print('Dev set evaluation by seen queries (full graph)')
        src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.all_objects, seen_queries, verbose=True)
    elif args.abs2real_infer:
        #--use_action_space_bucketing
        print ("CHECK abs verbose True")
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        test_path = os.path.join(args.data_dir, 'test.triples')
        dev_data = data_utils.load_triples(
            dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
        # print("======>CUTOFF dev_data210")
        # dev_data = dev_data[:20]

        dev_data_abs = data_utils.convert_entities2typeids(dev_data, entity2typeid)
        test_data = data_utils.load_triples(
            test_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
        test_data_abs = data_utils.convert_entities2typeids(test_data, entity2typeid)
        print('Dev set performance(lf.forward abs_graph=True):')
        # from src.rl.graph_search.beam_search import ABS_ALL_PATH

        from src.rl.graph_search.beam_search import ABS_ALL_PATH
        # print("read abs_path_dir:", args.abs_path_dir)

        def get_next_outs(graph, r, tid):
            ans = []
            r = int(r)
            if r not in graph:
                return ans  # TODO:CHECK 具体graph确实可能没有抽象r
            for e2 in graph[r]:
                if lf.kg.get_typeid(e2) == tid:
                    ans.append(e2)
            return ans

        def abs2real_path(abs_traces, data, k=args.beam_size):
            # data = data.cpu().numpy()
            tot_paths = []
            for abs_trace in abs_traces:  # 一个abs_trace代表一个样本
                pred_e2s = abs_trace['pred_e2s'][0]
                pred_e2_scores = torch.exp( abs_trace['pred_e2_scores'][0])
                search_traces = abs_trace['search_traces']
                paths = [[] for _ in range(len(search_traces[0][0]))]  # 一个样本所有roll_out出来的path,假设128

                for step in range(len(search_traces)):
                    for _i in range(len(search_traces[step][0])):
                        paths[_i].append((search_traces[step][0][_i], search_traces[step][1][_i]))

                for _i in range(len(paths)):
                    paths[_i].append((pred_e2s[_i], pred_e2_scores[_i]))

                tot_paths.append(paths)  # 认为对应一条样本
                # one path:  [(r_0, e_0), (r_1, e_1), (r_2, e_2), (r_3,e_3), (e_3, prob)]
            # print("paths[-1]:", paths[-1])
            # search path
            score_rslts = []
            missing = [0] * 10
            for i, paths in enumerate(tot_paths):  # paths代表真实一个样本
                tree_path = collections.defaultdict()
                e1_0, e2, r_0 = data[i]  # [e1,e2, r]已经是id了 这里假设e2不是list

                score_mat = zeros_var_cuda([1, lf.kg.num_entities])
                real_e3_score = []
                for path in paths:  # 一条样本的一个path
                    if lf.kg.get_typeid(e1_0) != path[0][1]:
                        print(" ERROR e1 != path[0][1]:", e1_0, path[0], " idx:", i)
                        missing[0] += 1
                        continue
                    if e1_0 in lf.kg.adj_list:
                        tree_path[e1_0] = collections.defaultdict()
                        # get_next_outs(lf.kg.adjlist)
                        # if r_0 not in lf.kg.adj_list[e1_0]:
                        #     missing[1] += 1
                        #     continue
                        e1_1s = get_next_outs(lf.kg.adj_list[e1_0], path[1][0], path[1][1])
                        if len(e1_1s) == 0:
                            missing[2] += 1
                            continue
                        e1_2s = []
                        for e1_1 in e1_1s:
                            outs = get_next_outs(lf.kg.adj_list[e1_1], path[2][0], path[2][1])
                            e1_2s += outs
                        if len(e1_2s) == 0:
                            missing[3] += 1
                            continue
                        e1_3s = []
                        for e1_2 in e1_2s:
                            outs = get_next_outs(lf.kg.adj_list[e1_2], path[3][0], path[3][1])
                            e1_3s += outs
                        if len(e1_3s) == 0:
                            missing[4] += 1
                            continue
                        # type2score = dict(zip())

                        for e1_3 in e1_3s:
                            # try:
                            #     path[-1][1] = path[-1][1].cpu().numpy()
                            # except:
                            #     pass
                            # real_e3_score.append((e1_3, np.exp(path[-1][1])))
                            real_e3_score.append((e1_3, path[-1][1]))
                    else:
                        print("ERROR e not on real KG:", e1_0)
                for e3, p in sorted(real_e3_score, key=lambda d: d[1], reverse=True)[:k]:
                    if p > float(score_mat[0, e3]):
                        score_mat[0, e3] = p
                    # score_mat[0, e3] += p  # TODO:CHECK
                score_rslts.append(score_mat)
            if len(score_rslts) == 0:
                print("MAYBE ERROR: len(score_rslts) == 0")
                score_rslts = zeros_var_cuda([len(data), lf.kg.num_entities])
            else:
                score_rslts = torch.cat(score_rslts)  # [nsample, num_entities]
            print("missing:", missing)
            return score_rslts

        global ABS_ALL_PATH
        CHECK = False
        # CHECK2 = True
        if not CHECK:
            pred_scores = lf.forward(dev_data_abs, abs_graph=True, verbose=False)
            dev_metrics = src.eval.hits_and_ranks(dev_data_abs, pred_scores, lf.kg.dev_objects_abs, verbose=True)
            eval_metrics['dev'] = {}
            eval_metrics['dev']['hits_at_1'] = dev_metrics[0]
            eval_metrics['dev']['hits_at_3'] = dev_metrics[1]
            eval_metrics['dev']['hits_at_5'] = dev_metrics[2]
            eval_metrics['dev']['hits_at_10'] = dev_metrics[3]
            eval_metrics['dev']['mrr'] = dev_metrics[4]
            src.eval.hits_and_ranks(dev_data_abs, pred_scores, lf.kg.all_objects_abs, verbose=True)
            print('Dev set performance(lf.forward abs_graph=True):')  # TODO: check一下这个hits_and_ranks是否适用abs
            abs_traces = ABS_ALL_PATH
        else:
            abs_traces = pickle.load(open('path2check_0.pkl', 'rb'))
            abs_traces = [abs_traces, ]  # 因为现在是一个样本一个Batch...
            dev_data = [dev_data[0], ]
        print("ALL PATH CNT:", len(abs_traces), len(dev_data))
        pred_scores = abs2real_path(abs_traces, dev_data)  # TODO:CHECK下面是不是lf.kg.all_objects_abs
        print("====> abs2real_path pred_scores shape:", pred_scores.shape)
        dev_metrics = src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.dev_objects, verbose=True)
        eval_metrics['dev_real'] = {}
        eval_metrics['dev_real']['hits_at_1'] = dev_metrics[0]
        eval_metrics['dev_real']['hits_at_3'] = dev_metrics[1]
        eval_metrics['dev_real']['hits_at_5'] = dev_metrics[2]
        eval_metrics['dev_real']['hits_at_10'] = dev_metrics[3]
        eval_metrics['dev_real']['mrr'] = dev_metrics[4]
        src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.all_objects, verbose=True)
        print('Dev set performance real (lf.forward abs_graph=True):')  # TODO: check一下这个hits_and_ranks是否适用abs

        ABS_ALL_PATH = []

        pred_scores = lf.forward(test_data_abs, abs_graph=True, verbose=False)
        test_metrics = src.eval.hits_and_ranks(test_data_abs, pred_scores, lf.kg.all_objects_abs,
                                               verbose=True)  # TODO: check一下这个hits_and_ranks是否适用abs
        eval_metrics['test']['hits_at_1'] = test_metrics[0]
        eval_metrics['test']['hits_at_3'] = test_metrics[1]
        eval_metrics['test']['hits_at_5'] = test_metrics[2]
        eval_metrics['test']['hits_at_10'] = test_metrics[3]
        eval_metrics['test']['mrr'] = test_metrics[4]
        print('Test set performance abs (lf.forward abs_graph=True):', eval_metrics)

        from src.rl.graph_search.beam_search import ABS_ALL_PATH
        # global ABS_ALL_PATH
        abs_traces = ABS_ALL_PATH[len(dev_data):]
        print("TEST ALL PATH CNT:", len(abs_traces), len(test_data))
        pred_scores = abs2real_path(abs_traces, test_data)
        test_metrics = src.eval.hits_and_ranks(test_data, pred_scores, lf.kg.all_objects,
                                               verbose=True)  # TODO: check一下这个hits_and_ranks是否适用abs
        eval_metrics['test']['hits_at_1'] = test_metrics[0]
        eval_metrics['test']['hits_at_3'] = test_metrics[1]
        eval_metrics['test']['hits_at_5'] = test_metrics[2]
        eval_metrics['test']['hits_at_10'] = test_metrics[3]
        eval_metrics['test']['mrr'] = test_metrics[4]
        print('Test set performance real (lf.forward abs_graph=True):', eval_metrics)


    elif args.same_infer:
        CHECK = False
        print ("CHECK same_infer verbose True, CHECK:{}".format(CHECK))
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        if CHECK:
            dev_path = dev_path[:100]
        test_path = os.path.join(args.data_dir, 'test.triples')
        dev_data = data_utils.load_triples(
            dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
        test_data = data_utils.load_triples(
            test_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
        print('Dev set performance:')
        # pred_scores = lf.forward(dev_data, same_infer=True, verbose=False)
        dev_scores = lf.forward(dev_data, verbose=False)
        # print ("dumping...")
        # open('check_pred_scores.pkl', 'wb').write(pickle.dumps(pred_scores))

        dev_metrics = src.eval.hits_and_ranks(dev_data, dev_scores, lf.kg.dev_objects, verbose=True)
        eval_metrics['dev'] = {}
        eval_metrics['dev']['hits_at_1'] = dev_metrics[0]
        eval_metrics['dev']['hits_at_3'] = dev_metrics[1]
        eval_metrics['dev']['hits_at_5'] = dev_metrics[2]
        eval_metrics['dev']['hits_at_10'] = dev_metrics[3]
        eval_metrics['dev']['mrr'] = dev_metrics[4]
        print('Dev set performance: baseline (include test set labels)')
        src.eval.hits_and_ranks(dev_data, dev_scores, lf.kg.all_objects, verbose=True)

        print('Dev set performance: abs (include test set labels)')
        dev_scores_abs2real = lf.forward(dev_data, verbose=False, same_infer=True)
        src.eval.hits_and_ranks(dev_data, dev_scores_abs2real, lf.kg.all_objects, verbose=True)

        dev_scores_force_merge = args.merge_abs_real_score * dev_scores + (
                                                                              1.0 - args.merge_abs_real_score) * dev_scores_abs2real

        print(
            'Dev set performance of abs model force_merge: (include test set labels)')
        src.eval.hits_and_ranks(
            dev_data, dev_scores_force_merge, lf.kg.all_objects, verbose=True)

        print(
            'Dev set performance of abs model force_merge same type inner: (include test set labels)')
        src.eval.hits_and_ranks_merge_inner(
            dev_data, dev_scores, lf.kg.all_objects, dev_scores_abs2real, lf.merge_abs_real_score,
            lf.kg.entity2typeid, verbose=True)



        print('Test set performance:')
        pred_scores = lf.forward(test_data,  verbose=False)
        test_metrics = src.eval.hits_and_ranks(test_data, pred_scores, lf.kg.all_objects, verbose=True)
        eval_metrics['test']['hits_at_1'] = test_metrics[0]
        eval_metrics['test']['hits_at_3'] = test_metrics[1]
        eval_metrics['test']['hits_at_5'] = test_metrics[2]
        eval_metrics['test']['hits_at_10'] = test_metrics[3]
        eval_metrics['test']['mrr'] = test_metrics[4]

        test_scores_abs2real = lf.forward(test_data, verbose=False, same_infer=True)
        # print(
        #     'Test set performance of abs model on ori graph: (correct evaluation)')
        # _, _, _, _, mrr = src.eval.hits_and_ranks(
        #     test_data, test_scores_abs2real, lf.kg.all_objects, verbose=True)
        # metrics = mrr
        print(
            'Test set performance of abs model on ori graph: (include test set labels)')
        src.eval.hits_and_ranks(
            test_data, test_scores_abs2real, lf.kg.all_objects, verbose=True)

        # merge
        test_scores_force_merge = lf.merge_abs_real_score * pred_scores + (
                                                                              1.0 - args.merge_abs_real_score) * test_scores_abs2real

        # _, _, _, _, mrr = src.eval.hits_and_ranks(
        #     dev_data, dev_scores_force_merge, self.kg.dev_objects, verbose=True)
        print("=========================\n")
        # metrics = mrr
        print(
            'Test set performance of abs model force_merge: (include test set labels)')
        src.eval.hits_and_ranks(
            test_data, test_scores_force_merge, lf.kg.all_objects, verbose=True)

        # from src.rl.graph_search.beam_search import REAL_ALL_PATHS, SAME_ALL_PATHS
        # merge by path， 如果real_path和abs_real_path的type一样 就认为可以merge


        # _, _, _, _, mrr = src.eval.hits_and_ranks_merge(
        #     dev_data, dev_scores, self.kg.all_objects, dev_scores_abs2real, self.merge_abs_real_score, self.kg.entity2typeid, verbose=True)
        # metrics = mrr
        print("=========================\n")

        print(
            'Test set performance of abs model force_merge same type inner: (include test set labels)')
        src.eval.hits_and_ranks_merge_inner(
            test_data, pred_scores, lf.kg.all_objects, test_scores_abs2real, args.merge_abs_real_score,
            lf.kg.entity2typeid, verbose=True)


    elif args.use_abstract_graph:
        print ("CHECK abs verbose True")
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        test_path = os.path.join(args.data_dir, 'test.triples')
        dev_data = data_utils.load_triples(
            dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
        dev_data_abs = data_utils.convert_entities2typeids(dev_data, entity2typeid)
        test_data = data_utils.load_triples(
            test_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
        test_data_abs = data_utils.convert_entities2typeids(test_data, entity2typeid)
        print('Dev set performance(lf.forward abs_graph=True):')
        # from src.rl.graph_search.beam_search import ABS_ALL_PATH
        pred_scores = lf.forward(dev_data_abs, abs_graph=True, verbose=False)
        dev_metrics = src.eval.hits_and_ranks(dev_data_abs, pred_scores, lf.kg.dev_objects_abs, verbose=True)
        eval_metrics['dev'] = {}
        eval_metrics['dev']['hits_at_1'] = dev_metrics[0]
        eval_metrics['dev']['hits_at_3'] = dev_metrics[1]
        eval_metrics['dev']['hits_at_5'] = dev_metrics[2]
        eval_metrics['dev']['hits_at_10'] = dev_metrics[3]
        eval_metrics['dev']['mrr'] = dev_metrics[4]
        src.eval.hits_and_ranks(dev_data_abs, pred_scores, lf.kg.all_objects_abs, verbose=True)
        print('Test set performance(lf.forward abs_graph=True):')  # TODO: check一下这个hits_and_ranks是否适用abs
        pred_scores = lf.forward(test_data_abs, abs_graph=True, verbose=False)
        test_metrics = src.eval.hits_and_ranks(test_data_abs, pred_scores, lf.kg.all_objects_abs,
                                               verbose=True)  # TODO: check一下这个hits_and_ranks是否适用abs
        eval_metrics['test']['hits_at_1'] = test_metrics[0]
        eval_metrics['test']['hits_at_3'] = test_metrics[1]
        eval_metrics['test']['hits_at_5'] = test_metrics[2]
        eval_metrics['test']['hits_at_10'] = test_metrics[3]
        eval_metrics['test']['mrr'] = test_metrics[4]
    else:
        print ("CHECK verbose True")
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        test_path = os.path.join(args.data_dir, 'test.triples')
        dev_data = data_utils.load_triples(
            dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
        test_data = data_utils.load_triples(
            test_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
        print('Dev set performance:')
        pred_scores = lf.forward(dev_data, verbose=False)
        print ("dumping...")
        open('check_pred_scores.pkl', 'wb').write(pickle.dumps(pred_scores))

        dev_metrics = src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.dev_objects, verbose=True)
        eval_metrics['dev'] = {}
        eval_metrics['dev']['hits_at_1'] = dev_metrics[0]
        eval_metrics['dev']['hits_at_3'] = dev_metrics[1]
        eval_metrics['dev']['hits_at_5'] = dev_metrics[2]
        eval_metrics['dev']['hits_at_10'] = dev_metrics[3]
        eval_metrics['dev']['mrr'] = dev_metrics[4]
        src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.all_objects, verbose=True)
        print('Test set performance:')
        pred_scores = lf.forward(test_data, verbose=False)
        test_metrics = src.eval.hits_and_ranks(test_data, pred_scores, lf.kg.all_objects, verbose=True)
        eval_metrics['test']['hits_at_1'] = test_metrics[0]
        eval_metrics['test']['hits_at_3'] = test_metrics[1]
        eval_metrics['test']['hits_at_5'] = test_metrics[2]
        eval_metrics['test']['hits_at_10'] = test_metrics[3]
        eval_metrics['test']['mrr'] = test_metrics[4]

    return eval_metrics


def run_ablation_studies(args):
    """
    Run the ablation study experiments reported in the paper.
    """

    def set_up_lf_for_inference(args):
        initialize_model_directory(args)
        lf = construct_model(args)
        lf.cuda()
        lf.batch_size = args.dev_batch_size
        lf.load_checkpoint(get_checkpoint_path(args))
        lf.eval()
        return lf

    def rel_change(metrics, ab_system, kg_portion):
        ab_system_metrics = metrics[ab_system][kg_portion]
        base_metrics = metrics['ours'][kg_portion]
        return int(np.round((ab_system_metrics - base_metrics) / base_metrics * 100))

    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    if 'NELL' in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()
    dataset = os.path.basename(args.data_dir)
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    dev_data = data_utils.load_triples(
        dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
    to_m_rels, to_1_rels, (to_m_ratio, to_1_ratio) = data_utils.get_relations_by_type(args.data_dir,
                                                                                      relation_index_path)
    relation_by_types = (to_m_rels, to_1_rels)
    to_m_ratio *= 100
    to_1_ratio *= 100
    seen_queries, (seen_ratio, unseen_ratio) = data_utils.get_seen_queries(args.data_dir, entity_index_path,
                                                                           relation_index_path)
    seen_ratio *= 100
    unseen_ratio *= 100

    systems = ['ours', '-ad', '-rs']
    mrrs, to_m_mrrs, to_1_mrrs, seen_mrrs, unseen_mrrs = {}, {}, {}, {}, {}
    for system in systems:
        print('** Evaluating {} system **'.format(system))
        if system == '-ad':
            args.action_dropout_rate = 0.0
            if dataset == 'umls':
                # adjust dropout hyperparameters
                args.emb_dropout_rate = 0.3
                args.ff_dropout_rate = 0.1
        elif system == '-rs':
            config_path = os.path.join('configs', '{}.sh'.format(dataset.lower()))
            args = parser.parse_args()
            args = data_utils.load_configs(args, config_path)

        lf = set_up_lf_for_inference(args)
        pred_scores = lf.forward(dev_data, verbose=False)
        _, _, _, _, mrr = src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.dev_objects, verbose=True)
        if to_1_ratio == 0:
            to_m_mrr = mrr
            to_1_mrr = -1
        else:
            to_m_mrr, to_1_mrr = src.eval.hits_and_ranks_by_relation_type(
                dev_data, pred_scores, lf.kg.dev_objects, relation_by_types, verbose=True)
        seen_mrr, unseen_mrr = src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.dev_objects, seen_queries, verbose=True)
        mrrs[system] = {
            '': mrr * 100
        }
        to_m_mrrs[system] = {
            '': to_m_mrr * 100
        }
        to_1_mrrs[system] = {
            '': to_1_mrr * 100
        }
        seen_mrrs[system] = {
            '': seen_mrr * 100
        }
        unseen_mrrs[system] = {
            '': unseen_mrr * 100
        }
        _, _, _, _, mrr_full_kg = src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.all_objects, verbose=True)
        if to_1_ratio == 0:
            to_m_mrr_full_kg = mrr_full_kg
            to_1_mrr_full_kg = -1
        else:
            to_m_mrr_full_kg, to_1_mrr_full_kg = src.eval.hits_and_ranks_by_relation_type(
                dev_data, pred_scores, lf.kg.all_objects, relation_by_types, verbose=True)
        seen_mrr_full_kg, unseen_mrr_full_kg = src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.all_objects, seen_queries, verbose=True)
        mrrs[system]['full_kg'] = mrr_full_kg * 100
        to_m_mrrs[system]['full_kg'] = to_m_mrr_full_kg * 100
        to_1_mrrs[system]['full_kg'] = to_1_mrr_full_kg * 100
        seen_mrrs[system]['full_kg'] = seen_mrr_full_kg * 100
        unseen_mrrs[system]['full_kg'] = unseen_mrr_full_kg * 100

    # overall system comparison (table 3)
    print('Partial graph evaluation')
    print('--------------------------')
    print('Overall system performance')
    print('Ours(ConvE)\t-RS\t-AD')
    print('{:.1f}\t{:.1f}\t{:.1f}'.format(mrrs['ours'][''], mrrs['-rs'][''], mrrs['-ad']['']))
    print('--------------------------')
    # performance w.r.t. relation types (table 4, 6)
    print('Performance w.r.t. relation types')
    print('\tTo-many\t\t\t\tTo-one\t\t')
    print('%\tOurs\t-RS\t-AD\t%\tOurs\t-RS\t-AD')
    print('{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})\t{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})'.format(
        to_m_ratio, to_m_mrrs['ours'][''], to_m_mrrs['-rs'][''], rel_change(to_m_mrrs, '-rs', ''), to_m_mrrs['-ad'][''],
        rel_change(to_m_mrrs, '-ad', ''),
        to_1_ratio, to_1_mrrs['ours'][''], to_1_mrrs['-rs'][''], rel_change(to_1_mrrs, '-rs', ''), to_1_mrrs['-ad'][''],
        rel_change(to_1_mrrs, '-ad', '')))
    print('--------------------------')
    # performance w.r.t. seen queries (table 5, 7)
    print('Performance w.r.t. seen/unseen queries')
    print('\tSeen\t\t\t\tUnseen\t\t')
    print('%\tOurs\t-RS\t-AD\t%\tOurs\t-RS\t-AD')
    print('{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})\t{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})'.format(
        seen_ratio, seen_mrrs['ours'][''], seen_mrrs['-rs'][''], rel_change(seen_mrrs, '-rs', ''), seen_mrrs['-ad'][''],
        rel_change(seen_mrrs, '-ad', ''),
        unseen_ratio, unseen_mrrs['ours'][''], unseen_mrrs['-rs'][''], rel_change(unseen_mrrs, '-rs', ''),
        unseen_mrrs['-ad'][''], rel_change(unseen_mrrs, '-ad', '')))
    print()
    print('Full graph evaluation')
    print('--------------------------')
    print('Overall system performance')
    print('Ours(ConvE)\t-RS\t-AD')
    print('{:.1f}\t{:.1f}\t{:.1f}'.format(mrrs['ours']['full_kg'], mrrs['-rs']['full_kg'], mrrs['-ad']['full_kg']))
    print('--------------------------')
    print('Performance w.r.t. relation types')
    print('\tTo-many\t\t\t\tTo-one\t\t')
    print('%\tOurs\t-RS\t-AD\t%\tOurs\t-RS\t-AD')
    print('{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})\t{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})'.format(
        to_m_ratio, to_m_mrrs['ours']['full_kg'], to_m_mrrs['-rs']['full_kg'], rel_change(to_m_mrrs, '-rs', 'full_kg'),
        to_m_mrrs['-ad']['full_kg'], rel_change(to_m_mrrs, '-ad', 'full_kg'),
        to_1_ratio, to_1_mrrs['ours']['full_kg'], to_1_mrrs['-rs']['full_kg'], rel_change(to_1_mrrs, '-rs', 'full_kg'),
        to_1_mrrs['-ad']['full_kg'], rel_change(to_1_mrrs, '-ad', 'full_kg')))
    print('--------------------------')
    print('Performance w.r.t. seen/unseen queries')
    print('\tSeen\t\t\t\tUnseen\t\t')
    print('%\tOurs\t-RS\t-AD\t%\tOurs\t-RS\t-AD')
    print('{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})\t{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})'.format(
        seen_ratio, seen_mrrs['ours']['full_kg'], seen_mrrs['-rs']['full_kg'], rel_change(seen_mrrs, '-rs', 'full_kg'),
        seen_mrrs['-ad']['full_kg'], rel_change(seen_mrrs, '-ad', 'full_kg'),
        unseen_ratio, unseen_mrrs['ours']['full_kg'], unseen_mrrs['-rs']['full_kg'],
        rel_change(unseen_mrrs, '-rs', 'full_kg'), unseen_mrrs['-ad']['full_kg'],
        rel_change(unseen_mrrs, '-ad', 'full_kg')))


def export_to_embedding_projector(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_to_embedding_projector()


def export_reward_shaping_parameters(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_reward_shaping_parameters()


def export_fuzzy_facts(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_fuzzy_facts()


def export_error_cases(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.batch_size = args.dev_batch_size
    lf.eval()
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path)
    lf.load_checkpoint(get_checkpoint_path(args))
    print('Dev set performance:')
    pred_scores = lf.forward(dev_data, verbose=False)
    src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.dev_objects, verbose=True)
    src.eval.export_error_cases(dev_data, pred_scores, lf.kg.dev_objects, os.path.join(lf.model_dir, 'error_cases.pkl'))


def compute_fact_scores(lf):
    data_dir = args.data_dir
    train_path = os.path.join(data_dir, 'train.triples')
    dev_path = os.path.join(data_dir, 'dev.triples')
    test_path = os.path.join(data_dir, 'test.triples')
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    train_data = data_utils.load_triples(train_path, entity_index_path, relation_index_path)
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path)
    test_data = data_utils.load_triples(test_path, entity_index_path, relation_index_path)
    lf.eval()
    lf.load_checkpoint(get_checkpoint_path(args))
    train_scores = lf.forward_fact(train_data)
    dev_scores = lf.forward_fact(dev_data)
    test_scores = lf.forward_fact(test_data)

    print('Train set average fact score: {}'.format(float(train_scores.mean())))
    print('Dev set average fact score: {}'.format(float(dev_scores.mean())))
    print('Test set average fact score: {}'.format(float(test_scores.mean())))


def get_checkpoint_path(args):
    if not args.checkpoint_path:
        return os.path.join(args.model_dir, 'model_best.tar')
    else:
        return args.checkpoint_path


def load_configs(config_path):
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


def run_experiment(args):
    if args.test:
        if 'NELL' in args.data_dir:
            dataset = os.path.basename(args.data_dir)
            args.distmult_state_dict_path = data_utils.change_to_test_model_path(dataset, args.distmult_state_dict_path)
            args.complex_state_dict_path = data_utils.change_to_test_model_path(dataset, args.complex_state_dict_path)
            args.conve_state_dict_path = data_utils.change_to_test_model_path(dataset, args.conve_state_dict_path)
        args.data_dir += '.test'

    if args.process_data:

        # Process knowledge graph data

        process_data()
    else:
        with torch.set_grad_enabled(args.train or args.search_random_seed or args.grid_search):
            if args.search_random_seed:

                # Search for best random seed

                # search log file
                task = os.path.basename(os.path.normpath(args.data_dir))
                out_log = '{}.{}.rss'.format(task, args.model)
                o_f = open(out_log, 'w')

                print('** Search Random Seed **')
                o_f.write('** Search Random Seed **\n')
                o_f.close()
                num_runs = 5

                hits_at_1s = {}
                hits_at_10s = {}
                mrrs = {}
                mrrs_search = {}
                for i in range(num_runs):

                    o_f = open(out_log, 'a')

                    random_seed = random.randint(0, 1e16)
                    print("\nRandom seed = {}\n".format(random_seed))
                    o_f.write("\nRandom seed = {}\n\n".format(random_seed))
                    torch.manual_seed(random_seed)
                    torch.cuda.manual_seed_all(args, random_seed)
                    initialize_model_directory(args, random_seed)
                    lf = construct_model(args)
                    lf.cuda()
                    train(lf)
                    metrics = inference(lf)
                    hits_at_1s[random_seed] = metrics['test']['hits_at_1']
                    hits_at_10s[random_seed] = metrics['test']['hits_at_10']
                    mrrs[random_seed] = metrics['test']['mrr']
                    mrrs_search[random_seed] = metrics['dev']['mrr']
                    # print the results of the hyperparameter combinations searched so far
                    print('------------------------------------------')
                    print('Random Seed\t@1\t@10\tMRR')
                    for key in hits_at_1s:
                        print('{}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
                            key, hits_at_1s[key], hits_at_10s[key], mrrs[key]))
                    print('------------------------------------------')
                    o_f.write('------------------------------------------\n')
                    o_f.write('Random Seed\t@1\t@10\tMRR\n')
                    for key in hits_at_1s:
                        o_f.write('{}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(
                            key, hits_at_1s[key], hits_at_10s[key], mrrs[key]))
                    o_f.write('------------------------------------------\n')

                    # compute result variance
                    import numpy as np
                    hits_at_1s_ = list(hits_at_1s.values())
                    hits_at_10s_ = list(hits_at_10s.values())
                    mrrs_ = list(mrrs.values())
                    print('Hits@1 mean: {:.3f}\tstd: {:.6f}'.format(np.mean(hits_at_1s_), np.std(hits_at_1s_)))
                    print('Hits@10 mean: {:.3f}\tstd: {:.6f}'.format(np.mean(hits_at_10s_), np.std(hits_at_10s_)))
                    print('MRR mean: {:.3f}\tstd: {:.6f}'.format(np.mean(mrrs_), np.std(mrrs_)))
                    o_f.write('Hits@1 mean: {:.3f}\tstd: {:.6f}\n'.format(np.mean(hits_at_1s_), np.std(hits_at_1s_)))
                    o_f.write('Hits@10 mean: {:.3f}\tstd: {:.6f}\n'.format(np.mean(hits_at_10s_), np.std(hits_at_10s_)))
                    o_f.write('MRR mean: {:.3f}\tstd: {:.6f}\n'.format(np.mean(mrrs_), np.std(mrrs_)))
                    o_f.close()

                # find best random seed
                best_random_seed, best_mrr = sorted(mrrs_search.items(), key=lambda x: x[1], reverse=True)[0]
                print('* Best Random Seed = {}'.format(best_random_seed))
                print('* @1: {:.3f}\t@10: {:.3f}\tMRR: {:.3f}'.format(
                    hits_at_1s[best_random_seed],
                    hits_at_10s[best_random_seed],
                    mrrs[best_random_seed]))
                with open(out_log, 'a'):
                    o_f.write('* Best Random Seed = {}\n'.format(best_random_seed))
                    o_f.write('* @1: {:.3f}\t@10: {:.3f}\tMRR: {:.3f}\n'.format(
                        hits_at_1s[best_random_seed],
                        hits_at_10s[best_random_seed],
                        mrrs[best_random_seed])
                    )
                    o_f.close()

            elif args.grid_search:

                # Grid search

                # search log file
                task = os.path.basename(os.path.normpath(args.data_dir))
                out_log = '{}.{}.gs'.format(task, args.model)
                o_f = open(out_log, 'w')

                print("** Grid Search **")
                o_f.write("** Grid Search **\n")
                hyperparameters = args.tune.split(',')

                if args.tune == '' or len(hyperparameters) < 1:
                    print("No hyperparameter specified.")
                    sys.exit(0)

                grid = hp_range[hyperparameters[0]]
                for hp in hyperparameters[1:]:
                    grid = itertools.product(grid, hp_range[hp])

                hits_at_1s = {}
                hits_at_10s = {}
                mrrs = {}
                grid = list(grid)
                print('* {} hyperparameter combinations to try'.format(len(grid)))
                o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
                o_f.close()

                for i, grid_entry in enumerate(list(grid)):

                    o_f = open(out_log, 'a')

                    if not (type(grid_entry) is list or type(grid_entry) is list):
                        grid_entry = [grid_entry]
                    grid_entry = flatten(grid_entry)
                    print('* Hyperparameter Set {}:'.format(i))
                    o_f.write('* Hyperparameter Set {}:\n'.format(i))
                    signature = ''
                    for j in range(len(grid_entry)):
                        hp = hyperparameters[j]
                        value = grid_entry[j]
                        if hp == 'bandwidth':
                            setattr(args, hp, int(value))
                        else:
                            setattr(args, hp, float(value))
                        signature += ':{}'.format(value)
                        print('* {}: {}'.format(hp, value))
                    initialize_model_directory(args)
                    lf = construct_model(args)
                    lf.cuda()
                    train(lf)
                    metrics = inference(lf)
                    hits_at_1s[signature] = metrics['dev']['hits_at_1']
                    hits_at_10s[signature] = metrics['dev']['hits_at_10']
                    mrrs[signature] = metrics['dev']['mrr']
                    # print the results of the hyperparameter combinations searched so far
                    print('------------------------------------------')
                    print('Signature\t@1\t@10\tMRR')
                    for key in hits_at_1s:
                        print('{}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
                            key, hits_at_1s[key], hits_at_10s[key], mrrs[key]))
                    print('------------------------------------------\n')
                    o_f.write('------------------------------------------\n')
                    o_f.write('Signature\t@1\t@10\tMRR\n')
                    for key in hits_at_1s:
                        o_f.write('{}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(
                            key, hits_at_1s[key], hits_at_10s[key], mrrs[key]))
                    o_f.write('------------------------------------------\n')
                    # find best hyperparameter set
                    best_signature, best_mrr = sorted(mrrs.items(), key=lambda x: x[1], reverse=True)[0]
                    print('* best hyperparameter set')
                    o_f.write('* best hyperparameter set\n')
                    best_hp_values = best_signature.split(':')[1:]
                    for i, value in enumerate(best_hp_values):
                        hp_name = hyperparameters[i]
                        hp_value = best_hp_values[i]
                        print('* {}: {}'.format(hp_name, hp_value))
                    print('* @1: {:.3f}\t@10: {:.3f}\tMRR: {:.3f}'.format(
                        hits_at_1s[best_signature],
                        hits_at_10s[best_signature],
                        mrrs[best_signature]
                    ))
                    o_f.write('* @1: {:.3f}\t@10: {:.3f}\tMRR: {:.3f}\ns'.format(
                        hits_at_1s[best_signature],
                        hits_at_10s[best_signature],
                        mrrs[best_signature]
                    ))

                    o_f.close()

            elif args.run_ablation_studies:
                run_ablation_studies(args)
            else:
                print("DEBUG else inference or train")
                initialize_model_directory(args)
                lf = construct_model(args)
                lf.cuda()

                if args.train:
                    print("DEBUG else train")
                    train(lf)
                elif args.inference:
                    inference(lf)
                elif args.eval_by_relation_type:
                    inference(lf)
                elif args.eval_by_seen_queries:
                    inference(lf)
                elif args.export_to_embedding_projector:
                    export_to_embedding_projector(lf)
                elif args.export_reward_shaping_parameters:
                    export_reward_shaping_parameters(lf)
                elif args.compute_fact_scores:
                    compute_fact_scores(lf)
                elif args.export_fuzzy_facts:
                    export_fuzzy_facts(lf)
                elif args.export_error_cases:
                    export_error_cases(lf)


if __name__ == '__main__':
    run_experiment(args)
