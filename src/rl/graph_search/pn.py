# -*- coding: utf-8 -*-
"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Graph Search Policy Network.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils.ops as ops
from src.utils.ops import var_cuda, zeros_var_cuda


class GraphSearchPolicy(nn.Module):
    def __init__(self, kg, args):
        super(GraphSearchPolicy, self).__init__()
        self.model = args.model
        self.relation_only = args.relation_only

        self.use_abstract_graph = args.use_abstract_graph
        print("GraphSearchPolicy use_abstract_graph:", self.use_abstract_graph)
        self.history_dim = args.history_dim
        self.history_num_layers = args.history_num_layers
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        if self.relation_only:
            self.action_dim = args.relation_dim
        else:  # 应该是走了这个分支
            self.action_dim = args.entity_dim + args.relation_dim
        self.ff_dropout_rate = args.ff_dropout_rate
        self.rnn_dropout_rate = args.rnn_dropout_rate
        self.action_dropout_rate = args.action_dropout_rate

        self.xavier_initialization = args.xavier_initialization

        self.relation_only_in_path = args.relation_only_in_path
        self.path = None

        # Set policy network modules
        self.define_modules()
        self.initialize_modules()

        # Fact network modules
        self.fn = None
        self.fn_kg = None
        self.kg = kg

    def transit(self, e, obs, kg, use_action_space_bucketing=True, merge_aspace_batching_outcome=False):
        """
        Compute the next action distribution based on
            (a) the current node (entity) in KG and the query relation
            (b) action history representation
        :param e: agent location (node) at step t.
        :param obs: agent observation at step t.
            e_s: source node
            q: query relation
            e_t: target node
            last_step: If set, the agent is carrying out the last step.
            last_r: label of edge traversed in the previous step
            seen_nodes: notes seen on the paths
        :param kg: Knowledge graph environment.
        :param use_action_space_bucketing: If set, group the action space of different nodes 
            into buckets by their sizes.
        :param merge_aspace_batch_outcome: If set, merge the transition probability distribution
            generated of different action space bucket into a single batch.
        :return
            With aspace batching and without merging the outcomes:
                db_outcomes: (Dynamic Batch) (action_space, action_dist)
                    action_space: (Batch) padded possible action indices
                    action_dist: (Batch) distribution over actions.
                inv_offset: Indices to set the dynamic batching output back to the original order.
                entropy: (Batch) entropy of action distribution.
            Else:
                action_dist: (Batch) distribution over actions.
                entropy: (Batch) entropy of action distribution.
        """
        e_s, q, e_t, last_step, last_r, seen_nodes = obs

        # Representation of the current state (current node and other observations)
        Q = kg.get_relation_embeddings(q)
        H = self.path[-1][0][-1, :, :]
        if self.relation_only:
            X = torch.cat([H, Q], dim=-1)
        elif self.relation_only_in_path:
            E_s = kg.get_entity_embeddings(e_s)
            E = kg.get_entity_embeddings(e)
            X = torch.cat([E, H, E_s, Q], dim=-1)
        else:  # 应该走的这个分支
            E = kg.get_entity_embeddings(e)
            X = torch.cat([E, H, Q], dim=-1)

        # MLP
        X = self.W1(X)
        X = F.relu(X)
        X = self.W1Dropout(X)
        X = self.W2(X)
        X2 = self.W2Dropout(X)

        def policy_nn_fun(X2, action_space):
            (r_space, e_space), action_mask = action_space
            A = self.get_action_embedding((r_space, e_space), kg)
            action_dist = F.softmax(
                torch.squeeze(A @ torch.unsqueeze(X2, 2), 2) - (1 - action_mask) * ops.HUGE_INT, dim=-1)
            # action_dist = ops.weighted_softmax(torch.squeeze(A @ torch.unsqueeze(X2, 2), 2), action_mask)
            return action_dist, ops.entropy(action_dist)

        def pad_and_cat_action_space(action_spaces, inv_offset):
            db_r_space, db_e_space, db_action_mask = [], [], []
            for (r_space, e_space), action_mask in action_spaces:
                db_r_space.append(r_space)
                db_e_space.append(e_space)
                db_action_mask.append(action_mask)
            r_space = ops.pad_and_cat(db_r_space, padding_value=kg.dummy_r)[inv_offset]
            e_space = ops.pad_and_cat(db_e_space, padding_value=kg.dummy_e)[inv_offset]
            action_mask = ops.pad_and_cat(db_action_mask, padding_value=0)[inv_offset]
            action_space = ((r_space, e_space), action_mask)
            return action_space

        if use_action_space_bucketing:  # 走的这个分支
            """
            
            """
            db_outcomes = []
            entropy_list = []
            references = []
            db_action_spaces, db_references = self.get_action_space_in_buckets(e, obs, kg)
            for action_space_b, reference_b in zip(db_action_spaces, db_references):
                X2_b = X2[reference_b, :]
                action_dist_b, entropy_b = policy_nn_fun(X2_b, action_space_b)
                references.extend(reference_b)
                db_outcomes.append((action_space_b, action_dist_b))
                entropy_list.append(entropy_b)
            inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
            entropy = torch.cat(entropy_list, dim=0)[inv_offset]
            if merge_aspace_batching_outcome:
                db_action_dist = []
                for _, action_dist in db_outcomes:
                    db_action_dist.append(action_dist)
                action_space = pad_and_cat_action_space(db_action_spaces, inv_offset)
                action_dist = ops.pad_and_cat(db_action_dist, padding_value=0)[inv_offset]
                db_outcomes = [(action_space, action_dist)]
                inv_offset = None
        else:
            action_space = self.get_action_space(e, obs, kg)
            action_dist, entropy = policy_nn_fun(X2, action_space)
            db_outcomes = [(action_space, action_dist)]
            inv_offset = None

        return db_outcomes, inv_offset, entropy

    def transit_on_abs(self, e, obs, kg, use_action_space_bucketing=True, merge_aspace_batching_outcome=False):
        """
        Compute the next action distribution based on
            (a) the current node (entity) in KG and the query relation
            (b) action history representation
        :param e: agent location (node) at step t.
        :param obs: agent observation at step t.
            e_s: source node
            q: query relation
            e_t: target node
            last_step: If set, the agent is carrying out the last step.
            last_r: label of edge traversed in the previous step
            seen_nodes: notes seen on the paths
        :param kg: Knowledge graph environment.
        :param use_action_space_bucketing: If set, group the action space of different nodes
            into buckets by their sizes.
        :param merge_aspace_batch_outcome: If set, merge the transition probability distribution
            generated of different action space bucket into a single batch.
        :return
            With aspace batching and without merging the outcomes:
                db_outcomes: (Dynamic Batch) (action_space, action_dist)
                    action_space: (Batch) padded possible action indices
                    action_dist: (Batch) distribution over actions.
                inv_offset: Indices to set the dynamic batching output back to the original order.
                entropy: (Batch) entropy of action distribution.
            Else:
                action_dist: (Batch) distribution over actions.
                entropy: (Batch) entropy of action distribution.
        """
        e_s, q, e_t, last_step, last_r, seen_nodes = obs

        # Representation of the current state (current node and other observations)
        Q = kg.get_relation_abs_embeddings(q)
        H = self.path_abs[-1][0][-1, :, :]
        if self.relation_only:
            X = torch.cat([H, Q], dim=-1)
        elif self.relation_only_in_path:
            E_s = kg.get_entity_abs_embeddings(e_s)
            E = kg.get_entity_abs_embeddings(e)
            X = torch.cat([E, H, E_s, Q], dim=-1)
        else:  # 应该走的这个分支
            E = kg.get_entity_abs_embeddings(e)
            X = torch.cat([E, H, Q], dim=-1)

        # MLP
        X = self.W1(X)
        X = F.relu(X)
        X = self.W1Dropout(X)
        X = self.W2(X)
        X2 = self.W2Dropout(X)

        def policy_nn_fun(X2, action_space):
            (r_space, e_space), action_mask = action_space
            A = self.get_action_embedding((r_space, e_space), kg)
            action_dist = F.softmax(
                torch.squeeze(A @ torch.unsqueeze(X2, 2), 2) - (1 - action_mask) * ops.HUGE_INT, dim=-1)
            # action_dist = ops.weighted_softmax(torch.squeeze(A @ torch.unsqueeze(X2, 2), 2), action_mask)
            return action_dist, ops.entropy(action_dist)

        def policy_nn_fun_abs(X2, action_space):
            (r_space, e_space), action_mask = action_space
            A = self.get_action_embedding_abs((r_space, e_space), kg)
            action_dist = F.softmax(
                torch.squeeze(A @ torch.unsqueeze(X2, 2), 2) - (1 - action_mask) * ops.HUGE_INT, dim=-1)
            # action_dist = ops.weighted_softmax(torch.squeeze(A @ torch.unsqueeze(X2, 2), 2), action_mask)
            return action_dist, ops.entropy(action_dist)

        def pad_and_cat_action_space(action_spaces, inv_offset):
            db_r_space, db_e_space, db_action_mask = [], [], []
            for (r_space, e_space), action_mask in action_spaces:
                db_r_space.append(r_space)
                db_e_space.append(e_space)
                db_action_mask.append(action_mask)
            r_space = ops.pad_and_cat(db_r_space, padding_value=kg.dummy_r)[inv_offset]
            e_space = ops.pad_and_cat(db_e_space, padding_value=kg.dummy_e)[inv_offset]
            action_mask = ops.pad_and_cat(db_action_mask, padding_value=0)[inv_offset]
            action_space = ((r_space, e_space), action_mask)
            return action_space

        if use_action_space_bucketing:  # 走的这个分支
            """

            """
            db_outcomes = []
            entropy_list = []
            references = []
            db_action_spaces, db_references = self.get_action_space_in_buckets_on_abs(e, obs, kg)
            for action_space_b, reference_b in zip(db_action_spaces, db_references):
                X2_b = X2[reference_b, :]
                action_dist_b, entropy_b = policy_nn_fun_abs(X2_b, action_space_b)
                references.extend(reference_b)
                db_outcomes.append((action_space_b, action_dist_b))
                entropy_list.append(entropy_b)
            inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
            entropy = torch.cat(entropy_list, dim=0)[inv_offset]
            if merge_aspace_batching_outcome:
                db_action_dist = []
                for _, action_dist in db_outcomes:
                    db_action_dist.append(action_dist)
                action_space = pad_and_cat_action_space(db_action_spaces, inv_offset)
                action_dist = ops.pad_and_cat(db_action_dist, padding_value=0)[inv_offset]
                db_outcomes = [(action_space, action_dist)]
                inv_offset = None
        else:
            action_space = self.get_action_space_abs(e, obs, kg)
            action_dist, entropy = policy_nn_fun(X2, action_space)
            db_outcomes = [(action_space, action_dist)]
            inv_offset = None

        return db_outcomes, inv_offset, entropy

    def transit_with_abs(self, e, obs, e_abs, obs_abs, kg, use_action_space_bucketing=True,
                         merge_aspace_batching_outcome=False):
        """
        Compute the next action distribution based on
            (a) the current node (entity) in KG and the query relation
            (b) action history representation
        :param e: agent location (node) at step t.
        :param obs: agent observation at step t.
            e_s: source node
            q: query relation
            e_t: target node
            last_step: If set, the agent is carrying out the last step.
            last_r: label of edge traversed in the previous step
            seen_nodes: notes seen on the paths
        :param kg: Knowledge graph environment.
        :param use_action_space_bucketing: If set, group the action space of different nodes
            into buckets by their sizes.
        :param merge_aspace_batch_outcome: If set, merge the transition probability distribution
            generated of different action space bucket into a single batch.
        :return
            With aspace batching and without merging the outcomes:
                db_outcomes: (Dynamic Batch) (action_space, action_dist)
                    action_space: (Batch) padded possible action indices
                    action_dist: (Batch) distribution over actions.
                inv_offset: Indices to set the dynamic batching output back to the original order.
                entropy: (Batch) entropy of action distribution.
            Else:
                action_dist: (Batch) distribution over actions.
                entropy: (Batch) entropy of action distribution.
        """
        e_s, q, e_t, last_step, last_r, seen_nodes = obs
        e_s_abs, q_abs, e_t_abs, last_step_abs, last_r_abs, seen_nodes_abs = obs_abs

        # Representation of the current state (current node and other observations)
        Q = kg.get_relation_embeddings(q)
        Q_abs = kg.get_relation_abs_embeddings(q_abs)
        H = self.path[-1][0][-1, :, :]
        H_abs = self.path_abs[-1][0][-1, :, :]
        if self.relation_only:
            X = torch.cat([H, Q], dim=-1)
            X_abs = torch.cat([H_abs, Q_abs], dim=-1)
        elif self.relation_only_in_path:
            E_s = kg.get_entity_embeddings(e_s)
            E_s_abs = kg.get_entity_abs_embeddings(e_s_abs)
            E = kg.get_entity_embeddings(e)
            E_abs = kg.get_entity_abs_embeddings(e_abs)
            X = torch.cat([E, H, E_s, Q], dim=-1)
            X_abs = torch.cat([E_abs, H_abs, E_s_abs, Q_abs], dim=-1)
        else:  # 应该走的这个分支
            E = kg.get_entity_embeddings(e)
            E_abs = kg.get_entity_abs_embeddings(e_abs)
            X = torch.cat([E, H, Q], dim=-1)
            X_abs = torch.cat([E_abs, H_abs, Q_abs], dim=-1)

        # MLP
        # X = self.W1(X)
        # X = F.relu(X)
        # X = self.W1Dropout(X)
        # X = self.W2(X)
        # X2 = self.W2Dropout(X)
        def MLP(X, W1, W1DP, W2, W2DP):
            X = W1DP(F.relu(W1(X)))
            X = W2DP(W2(X))
            return X

        X2 = MLP(X, self.W1, self.W1Dropout, self.W2, self.W2Dropout)
        X2_abs = MLP(X_abs, self.W1_abstract, self.W1Dropout_abstract, self.W2_abstract, self.W2Dropout_abstract)

        def policy_nn_fun(X2, action_space):
            (r_space, e_space), action_mask = action_space
            A = self.get_action_embedding((r_space, e_space), kg)
            action_dist = F.softmax(
                torch.squeeze(A @ torch.unsqueeze(X2, 2), 2) - (1 - action_mask) * ops.HUGE_INT, dim=-1)
            # action_dist = ops.weighted_softmax(torch.squeeze(A @ torch.unsqueeze(X2, 2), 2), action_mask)
            return action_dist, ops.entropy(action_dist)

        def policy_nn_fun_abs(X2, action_space):
            (r_space, e_space), action_mask = action_space
            A = self.get_action_embedding_abs((r_space, e_space), kg)
            action_dist = F.softmax(
                torch.squeeze(A @ torch.unsqueeze(X2, 2), 2) - (1 - action_mask) * ops.HUGE_INT, dim=-1)
            # action_dist = ops.weighted_softmax(torch.squeeze(A @ torch.unsqueeze(X2, 2), 2), action_mask)
            return action_dist, ops.entropy(action_dist)

        def pad_and_cat_action_space(action_spaces, inv_offset):
            db_r_space, db_e_space, db_action_mask = [], [], []
            for (r_space, e_space), action_mask in action_spaces:
                db_r_space.append(r_space)
                db_e_space.append(e_space)
                db_action_mask.append(action_mask)
            r_space = ops.pad_and_cat(db_r_space, padding_value=kg.dummy_r)[inv_offset]
            e_space = ops.pad_and_cat(db_e_space, padding_value=kg.dummy_e)[inv_offset]
            action_mask = ops.pad_and_cat(db_action_mask, padding_value=0)[inv_offset]
            action_space = ((r_space, e_space), action_mask)
            return action_space

        if use_action_space_bucketing:  # 走的这个分支
            """

            """
            db_outcomes = []
            db_outcomes_abs = []
            entropy_list = []
            entropy_list_abs = []
            references = []
            references_abs = []
            print ("in shape:", e.size(), e_abs.size())
            db_action_spaces, db_references, db_action_spaces_abs, db_references_abs = self.get_action_space_in_buckets_with_abs(
                e, obs, e_abs, obs_abs, kg)

            print("dba dbr dbaa dbar:", db_action_spaces[0][0][0].size(), db_action_spaces_abs[0][0][0].size())
            for action_space_b, reference_b in zip(db_action_spaces, db_references):
                X2_b = X2[reference_b, :]
                action_dist_b, entropy_b = policy_nn_fun(X2_b, action_space_b)
                references.extend(reference_b)
                db_outcomes.append((action_space_b, action_dist_b))
                entropy_list.append(entropy_b)
            for action_space_b_abs, reference_b_abs in zip(db_action_spaces_abs, db_references_abs):
                X2_b_abs = X2_abs[reference_b_abs, :]
                action_dist_b_abs, entropy_b_abs = policy_nn_fun_abs(X2_b_abs, action_space_b_abs)
                references_abs.extend(reference_b_abs)
                db_outcomes_abs.append((action_space_b_abs, action_dist_b_abs))
                entropy_list_abs.append(entropy_b_abs)

            inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
            inv_offset_abs = [i for i, _ in sorted(enumerate(references_abs), key=lambda x: x[1])]
            entropy = torch.cat(entropy_list, dim=0)[inv_offset]
            entropy_abs = torch.cat(entropy_list_abs, dim=0)[inv_offset_abs]
            if merge_aspace_batching_outcome:
                db_action_dist = []
                db_action_dist_abs = []
                for _, action_dist in db_outcomes:
                    db_action_dist.append(action_dist)
                for _, action_dist_abs in db_outcomes_abs:
                    db_action_dist_abs.append(action_dist_abs)
                action_space = pad_and_cat_action_space(db_action_spaces, inv_offset)
                action_space_abs = pad_and_cat_action_space(db_action_spaces_abs, inv_offset_abs)
                action_dist = ops.pad_and_cat(db_action_dist, padding_value=0)[inv_offset]
                action_dist_abs = ops.pad_and_cat(db_action_dist_abs, padding_value=0)[inv_offset_abs]
                db_outcomes = [(action_space, action_dist)]
                db_outcomes_abs = [(action_space_abs, action_dist_abs)]
                inv_offset = None
                inv_offset_abs = None
        else:
            # raise NotImplementedError
            action_space = self.get_action_space(e, obs, kg)
            action_dist, entropy = policy_nn_fun(X2, action_space)
            
            #action_space_abs = self.get_action_space_abs(e_abs, obs_abs, kg)
            #action_space_abs = self.get_action_space_abs(e_abs, obs_abs, kg)
            #action_space_abs = self.generate_action_space_abs(action_space, e_abs, obs_abs, kg)
            action_space_abs = self.get_action_space_e2t(e, obs_abs, kg)
            action_dist_abs, entropy_abs = policy_nn_fun_abs(X2_abs, action_space_abs)
            db_outcomes = [(action_space, action_dist)]
            db_outcomes_abs = [(action_space_abs, action_dist_abs)]
            inv_offset = None
            inv_offset_abs = None

        #print ("+++>>>db_outcomes db_outcomes_abs:", len(db_outcomes), len(db_outcomes_abs))
        return db_outcomes, inv_offset, entropy, db_outcomes_abs, inv_offset_abs, entropy_abs
    
    def generate_action_space_abs(self, action_space, e_abs, obs_abs, kg):
        # def emat2tmat(es):
        #     ret = []
        #     # print(es)
        #     # print(es.size())
        #     for _ in range(es.size()[0]):
        #         for e in es[_]:
        #             ret.append(self.kg.get_typeid(e))
        #     ret = np.array(ret)
            
        #     #print("ret {} e_space===>>{} type_space".format(len(es), len(set(ret))) )
        #     ret = var_cuda(torch.LongTensor(ret), requires_grad=False)
        #     ret = ret.view(es.size()[0], es.size()[1])
        #     return ret

        def filter_overlapping(r_space, type_space, action_mask):
            print("len of r_space", len(r_space))
            print("len of type_space", len(type_space))
            assert(len(r_space) == len(type_space))
            print("type_space")
            print(type_space)
            print("action_mask")
            print(action_mask)

            filter_r_space = []
            filter_type_space = []
            filter_action_mask = []
            print(len(r_space))
            print(len(r_space[0]))
            for i in range(len(r_space)):
                temp = set()
                for _ in range(len(r_space[0])):
                    #print((r_space[i, _].tolist(), type_space[i, _].tolist()))
                #重复action或者是paddingaction
                    # or action_mask[i, _] == 0:
                    if (r_space[i, _].tolist(), type_space[i, _].tolist()) in temp:
                        temp.add((self.kg.dummy_r, self.kg.get_typeid(self.kg.dummy_e)))
                        filter_action_mask.append(0)
                        filter_r_space.append(self.kg.dummy_r)
                        filter_type_space.append(
                            self.kg.get_typeid(self.kg.dummy_e))
                    else:
                        filter_r_space.append(r_space[i, _].tolist())
                        filter_type_space.append(type_space[i, _].tolist())
                        temp.add((r_space[i, _].tolist(),
                                  type_space[i, _].tolist()))
                        filter_action_mask.append(1)

            filter_r_space = var_cuda(
                torch.LongTensor(filter_r_space), requires_grad=False)
            filter_type_space = var_cuda(
                torch.LongTensor(filter_type_space), requires_grad=False)
            filter_action_mask = var_cuda(
                torch.LongTensor(filter_action_mask), requires_grad=False).float()

            filter_type_space = filter_type_space.view(e_space.size()[0], e_space.size()[1])
            filter_r_space = filter_r_space.view(r_space.size()[0], r_space.size()[1])
            filter_action_mask = filter_action_mask.view(action_mask.size()[0], action_mask.size()[1])
            print("filter_type_space")
            print(filter_type_space)
            print("filter_action_mask")
            print(filter_action_mask)
            print("filter_r_space")
            print(filter_r_space)
            return filter_r_space, filter_type_space, filter_action_mask
        
        (r_space, e_space), action_mask = action_space
        print(e_space)
        type_space = self.kg.get_type_mat(e_space)
        print(type_space)
        filter_r_space, filter_type_space, filter_action_mask = filter_overlapping(
            r_space, type_space, action_mask)
        return ((filter_r_space, filter_type_space), filter_action_mask)

    
    def initialize_path(self, init_action, kg):
        # [batch_size, action_dim]
        if self.relation_only_in_path:
            init_action_embedding = kg.get_relation_embeddings(init_action[0])
        else:
            init_action_embedding = self.get_action_embedding(init_action, kg)
        init_action_embedding.unsqueeze_(1)
        # [num_layers, batch_size, dim]
        init_h = zeros_var_cuda([self.history_num_layers, len(init_action_embedding), self.history_dim])
        init_c = zeros_var_cuda([self.history_num_layers, len(init_action_embedding), self.history_dim])
        self.path = [self.path_encoder(init_action_embedding, (init_h, init_c))[1]]

    def initialize_abs_path(self, init_action, kg):
        # [batch_size, action_dim]
        if self.relation_only_in_path:
            init_action_embedding_abs = kg.get_relation_embeddings(init_action[0])
        else:
            init_action_embedding_abs = self.get_action_embedding(init_action, kg)
        init_action_embedding_abs.unsqueeze_(1)
        # [num_layers, batch_size, dim]
        init_h_abs = zeros_var_cuda([self.history_num_layers, len(init_action_embedding_abs), self.history_dim])
        init_c_abs = zeros_var_cuda([self.history_num_layers, len(init_action_embedding_abs), self.history_dim])
        self.path_abs = [self.path_abs_encoder(init_action_embedding_abs, (init_h_abs, init_c_abs))[1]]

    def update_path(self, action, kg, offset=None):
        """
        Once an action was selected, update the action history.
        :param action (r, e): (Variable:batch) indices of the most recent action
            - r is the most recently traversed edge;
            - e is the destination entity.
        :param offset: (Variable:batch) if None, adjust path history with the given offset, used for search
        :param KG: Knowledge graph environment.
        """

        def offset_path_history(p, offset):
            for i, x in enumerate(p):
                if type(x) is tuple:
                    new_tuple = tuple([_x[:, offset, :] for _x in x])
                    p[i] = new_tuple
                else:
                    p[i] = x[offset, :]

        # update action history
        if self.relation_only_in_path:
            action_embedding = kg.get_relation_embeddings(action[0])
        else:
            action_embedding = self.get_action_embedding(action, kg)
        if offset is not None:
            offset_path_history(self.path, offset)

        self.path.append(self.path_encoder(action_embedding.unsqueeze(1), self.path[-1])[1])

    def update_path_abs(self, action, kg, offset=None):
        """
        Once an action was selected, update the action history.
        :param action (r, e): (Variable:batch) indices of the most recent action
            - r is the most recently traversed edge;
            - e is the destination entity.
        :param offset: (Variable:batch) if None, adjust path history with the given offset, used for search
        :param KG: Knowledge graph environment.
        """

        def offset_path_history(p, offset):
            for i, x in enumerate(p):
                if type(x) is tuple:
                    new_tuple = tuple([_x[:, offset, :] for _x in x])
                    p[i] = new_tuple
                else:
                    p[i] = x[offset, :]

        # update action history
        if self.relation_only_in_path:
            action_embedding = kg.get_relation_abs_embeddings(action[0])
        else:
            action_embedding = self.get_action_embedding_abs(action, kg)
        if offset is not None:
            offset_path_history(self.path_abs, offset)

        self.path_abs.append(self.path_abs_encoder(action_embedding.unsqueeze(1), self.path_abs[-1])[1])

    def get_action_space_in_buckets(self, e, obs, kg, collapse_entities=False):
        """
        To compute the search operation in batch, we group the action spaces of different states
        (i.e. the set of outgoing edges of different nodes) into buckets based on their sizes to
        save the memory consumption of paddings.

        For example, in large knowledge graphs, certain nodes may have thousands of outgoing
        edges while a long tail of nodes only have a small amount of outgoing edges. If a batch
        contains a node with 1000 outgoing edges while the rest of the nodes have a maximum of
        5 outgoing edges, we need to pad the action spaces of all nodes to 1000, which consumes
        lots of memory.

        With the bucketing approach, each bucket is padded separately. In this case the node
        with 1000 outgoing edges will be in its own bucket and the rest of the nodes will suffer
        little from padding the action space to 5.

        Once we grouped the action spaces in buckets, the policy network computation is carried
        out for every bucket iteratively. Once all the computation is done, we concatenate the
        results of all buckets and restore their original order in the batch. The computation
        outside the policy network module is thus unaffected.

        :return db_action_spaces:
            [((r_space_b0, r_space_b0), action_mask_b0),
             ((r_space_b1, r_space_b1), action_mask_b1),
             ...
             ((r_space_bn, r_space_bn), action_mask_bn)]

            A list of action space tensor representations grouped in n buckets, s.t.
            r_space_b0.size(0) + r_space_b1.size(0) + ... + r_space_bn.size(0) = e.size(0)

        :return db_references:
            [l_batch_refs0, l_batch_refs1, ..., l_batch_refsn]
            l_batch_refsi stores the indices of the examples in bucket i in the current batch,
            which is used later to restore the output results to the original order.
        """
        e_s, q, e_t, last_step, last_r, seen_nodes = obs
        assert (len(e) == len(last_r))
        assert (len(e) == len(e_s))
        assert (len(e) == len(q))
        assert (len(e) == len(e_t))
        db_action_spaces, db_references = [], []

        if collapse_entities:
            raise NotImplementedError
        else:
            entity2bucketid = kg.entity2bucketid[e.tolist()]
            key1 = entity2bucketid[:, 0]
            key2 = entity2bucketid[:, 1]
            batch_ref = {}
            for i in range(len(e)):
                key = int(key1[i])
                if not key in batch_ref:
                    batch_ref[key] = []
                batch_ref[key].append(i)
            for key in batch_ref:
                action_space = kg.action_space_buckets[key]
                # l_batch_refs: ids of the examples in the current batch of examples
                # g_bucket_ids: ids of the examples in the corresponding KG action space bucket
                l_batch_refs = batch_ref[key]
                g_bucket_ids = key2[l_batch_refs].tolist()
                r_space_b = action_space[0][0][g_bucket_ids]
                e_space_b = action_space[0][1][g_bucket_ids]
                action_mask_b = action_space[1][g_bucket_ids]
                e_b = e[l_batch_refs]
                last_r_b = last_r[l_batch_refs]
                e_s_b = e_s[l_batch_refs]
                q_b = q[l_batch_refs]
                e_t_b = e_t[l_batch_refs]
                seen_nodes_b = seen_nodes[l_batch_refs]
                obs_b = [e_s_b, q_b, e_t_b, last_step, last_r_b, seen_nodes_b]
                action_space_b = ((r_space_b, e_space_b), action_mask_b)
                action_space_b = self.apply_action_masks(action_space_b, e_b, obs_b, kg)
                db_action_spaces.append(action_space_b)
                db_references.append(l_batch_refs)

        return db_action_spaces, db_references

    def get_action_space_in_buckets_on_abs(self, e, obs, kg, collapse_entities=False):
        """
        To compute the search operation in batch, we group the action spaces of different states
        (i.e. the set of outgoing edges of different nodes) into buckets based on their sizes to
        save the memory consumption of paddings.

        For example, in large knowledge graphs, certain nodes may have thousands of outgoing
        edges while a long tail of nodes only have a small amount of outgoing edges. If a batch
        contains a node with 1000 outgoing edges while the rest of the nodes have a maximum of
        5 outgoing edges, we need to pad the action spaces of all nodes to 1000, which consumes
        lots of memory.

        With the bucketing approach, each bucket is padded separately. In this case the node
        with 1000 outgoing edges will be in its own bucket and the rest of the nodes will suffer
        little from padding the action space to 5.

        Once we grouped the action spaces in buckets, the policy network computation is carried
        out for every bucket iteratively. Once all the computation is done, we concatenate the
        results of all buckets and restore their original order in the batch. The computation
        outside the policy network module is thus unaffected.

        :return db_action_spaces:
            [((r_space_b0, r_space_b0), action_mask_b0),
             ((r_space_b1, r_space_b1), action_mask_b1),
             ...
             ((r_space_bn, r_space_bn), action_mask_bn)]

            A list of action space tensor representations grouped in n buckets, s.t.
            r_space_b0.size(0) + r_space_b1.size(0) + ... + r_space_bn.size(0) = e.size(0)

        :return db_references:
            [l_batch_refs0, l_batch_refs1, ..., l_batch_refsn]
            l_batch_refsi stores the indices of the examples in bucket i in the current batch,
            which is used later to restore the output results to the original order.
        """
        e_s, q, e_t, last_step, last_r, seen_nodes = obs
        assert (len(e) == len(last_r))
        assert (len(e) == len(e_s))
        assert (len(e) == len(q))
        assert (len(e) == len(e_t))
        db_action_spaces, db_references = [], []

        if collapse_entities:
            raise NotImplementedError
        else:
            entity2bucketid = kg.entityabs2bucketid[e.tolist()]
            key1 = entity2bucketid[:, 0]
            key2 = entity2bucketid[:, 1]
            batch_ref = {}
            for i in range(len(e)):
                key = int(key1[i])
                if not key in batch_ref:
                    batch_ref[key] = []
                batch_ref[key].append(i)
            for key in batch_ref:
                action_space = kg.action_space_abs_buckets[key]
                # l_batch_refs: ids of the examples in the current batch of examples
                # g_bucket_ids: ids of the examples in the corresponding KG action space bucket
                l_batch_refs = batch_ref[key]
                g_bucket_ids = key2[l_batch_refs].tolist()
                r_space_b = action_space[0][0][g_bucket_ids]
                e_space_b = action_space[0][1][g_bucket_ids]
                action_mask_b = action_space[1][g_bucket_ids]
                e_b = e[l_batch_refs]
                last_r_b = last_r[l_batch_refs]
                e_s_b = e_s[l_batch_refs]
                q_b = q[l_batch_refs]
                e_t_b = e_t[l_batch_refs]
                seen_nodes_b = seen_nodes[l_batch_refs]
                obs_b = [e_s_b, q_b, e_t_b, last_step, last_r_b, seen_nodes_b]
                action_space_b = ((r_space_b, e_space_b), action_mask_b)
                action_space_b = self.apply_action_masks(action_space_b, e_b, obs_b, kg)
                db_action_spaces.append(action_space_b)
                db_references.append(l_batch_refs)

        return db_action_spaces, db_references

    def get_action_space_in_buckets_with_abs(self, e, obs, e_abs, obs_abs, kg, collapse_entities=False):
        """
        To compute the search operation in batch, we group the action spaces of different states
        (i.e. the set of outgoing edges of different nodes) into buckets based on their sizes to
        save the memory consumption of paddings.

        For example, in large knowledge graphs, certain nodes may have thousands of outgoing
        edges while a long tail of nodes only have a small amount of outgoing edges. If a batch
        contains a node with 1000 outgoing edges while the rest of the nodes have a maximum of
        5 outgoing edges, we need to pad the action spaces of all nodes to 1000, which consumes
        lots of memory.

        With the bucketing approach, each bucket is padded separately. In this case the node
        with 1000 outgoing edges will be in its own bucket and the rest of the nodes will suffer
        little from padding the action space to 5.

        Once we grouped the action spaces in buckets, the policy network computation is carried
        out for every bucket iteratively. Once all the computation is done, we concatenate the
        results of all buckets and restore their original order in the batch. The computation
        outside the policy network module is thus unaffected.

        :return db_action_spaces:
            [((r_space_b0, r_space_b0), action_mask_b0),
             ((r_space_b1, r_space_b1), action_mask_b1),
             ...
             ((r_space_bn, r_space_bn), action_mask_bn)]

            A list of action space tensor representations grouped in n buckets, s.t.
            r_space_b0.size(0) + r_space_b1.size(0) + ... + r_space_bn.size(0) = e.size(0)

        :return db_references:
            [l_batch_refs0, l_batch_refs1, ..., l_batch_refsn]
            l_batch_refsi stores the indices of the examples in bucket i in the current batch,
            which is used later to restore the output results to the original order.
        """
        e_s, q, e_t, last_step, last_r, seen_nodes = obs
        e_s_abs, q_abs, e_t_abs, last_step_abs, last_r_abs, seen_nodes_abs = obs_abs
        assert (len(e) == len(last_r))
        assert (len(e) == len(e_s))
        assert (len(e) == len(q))
        assert (len(e) == len(e_t))
        assert (len(e_abs) == len(last_r_abs))
        assert (len(e_abs) == len(e_s_abs))
        assert (len(e_abs) == len(q_abs))
        assert (len(e_abs) == len(e_t_abs))
        db_action_spaces, db_references = [], []
        db_action_spaces_abs, db_references_abs = [], []

        if collapse_entities:
            raise NotImplementedError
        else:
            entity2bucketid = kg.entity2bucketid[e.tolist()]
            entityabs2bucketid = kg.entityabs2bucketid[e_abs.tolist()]
            key1 = entity2bucketid[:, 0]
            key1_abs = entityabs2bucketid[:, 0]
            key2 = entity2bucketid[:, 1]
            key2_abs = entityabs2bucketid[:, 1]


            batch_ref = {}
            batch_ref_abs = {}
            for i in range(len(e)):
                key = int(key1[i])
                # key_abs = int(key1_abs[i])
                if not key in batch_ref:
                    batch_ref[key] = []
                # if not key_abs in batch_ref_abs:
                #     batch_ref_abs[key_abs] = []
                batch_ref[key].append(i)
                # batch_ref_abs[key_abs].append(i)
            for i in range(len(e_abs)):
                key_abs = int(key1_abs[i])
                if not key_abs in batch_ref_abs:
                    batch_ref_abs[key_abs] = []
                batch_ref_abs[key_abs].append(i)

            print("++++>bucket_size:", len(key1), "abs bucket_size:", len(key1_abs), "bucket_ref:", len(batch_ref), "batch_ref_abs:", len(batch_ref_abs))

            for key in batch_ref:
                action_space = kg.action_space_buckets[key]  # 从key的所有出边(r, e2)构成action_space
                # l_batch_refs: ids of the examples in the current batch of examples
                # g_bucket_ids: ids of the examples in the corresponding KG action space bucket
                l_batch_refs = batch_ref[key]
                g_bucket_ids = key2[l_batch_refs].tolist()
                r_space_b = action_space[0][0][g_bucket_ids]
                e_space_b = action_space[0][1][g_bucket_ids]
                action_mask_b = action_space[1][g_bucket_ids]
                e_b = e[l_batch_refs]
                last_r_b = last_r[l_batch_refs]
                e_s_b = e_s[l_batch_refs]
                q_b = q[l_batch_refs]
                e_t_b = e_t[l_batch_refs]
                seen_nodes_b = seen_nodes[l_batch_refs]
                obs_b = [e_s_b, q_b, e_t_b, last_step, last_r_b, seen_nodes_b]
                action_space_b = ((r_space_b, e_space_b), action_mask_b)
                action_space_b = self.apply_action_masks(action_space_b, e_b, obs_b, kg)
                db_action_spaces.append(action_space_b)
                db_references.append(l_batch_refs)
            for key in batch_ref_abs:
                action_space_abs = kg.action_space_abs_buckets[key]
                # l_batch_refs: ids of the examples in the current batch of examples
                # g_bucket_ids: ids of the examples in the corresponding KG action space bucket
                l_batch_refs = batch_ref_abs[key]
                g_bucket_ids = key2_abs[l_batch_refs].tolist()
                r_space_b = action_space_abs[0][0][g_bucket_ids]
                e_space_b = action_space_abs[0][1][g_bucket_ids]
                action_mask_b = action_space_abs[1][g_bucket_ids]
                e_b = e_abs[l_batch_refs]
                last_r_b = last_r_abs[l_batch_refs]
                e_s_b = e_s_abs[l_batch_refs]
                q_b = q_abs[l_batch_refs]
                e_t_b = e_t_abs[l_batch_refs]
                seen_nodes_b = seen_nodes_abs[l_batch_refs]
                obs_b = [e_s_b, q_b, e_t_b, last_step, last_r_b, seen_nodes_b]
                action_space_b = ((r_space_b, e_space_b), action_mask_b)
                action_space_b = self.apply_action_masks(action_space_b, e_b, obs_b, kg)  # TODO:CHECK 这里是不是可以共用
                db_action_spaces_abs.append(action_space_b)
                db_references_abs.append(l_batch_refs)

        return db_action_spaces, db_references, db_action_spaces_abs, db_references_abs

    def get_action_space(self, e, obs, kg):
        r_space, e_space = kg.action_space[0][0][e], kg.action_space[0][1][e]
        action_mask = kg.action_space[1][e]
        action_space = ((r_space, e_space), action_mask)
        return self.apply_action_masks(action_space, e, obs, kg)

    def get_action_space_abs(self, e, obs, kg):
        r_space, e_space = kg.action_space_abs[0][0][e], kg.action_space_abs[0][1][e]
        action_mask = kg.action_space_abs[1][e]
        action_space = ((r_space, e_space), action_mask)
        return self.apply_action_masks(action_space, e, obs, kg)

    def get_action_space_e2t(self, e, obs, kg):
        r_space, e_space = kg.action_space_e2t[0][0][e], kg.action_space_e2t[0][1][e]
        action_mask = kg.action_space_e2t[1][e]
        action_space = ((r_space, e_space), action_mask)
        return self.apply_action_masks(action_space, e, obs, kg)

    def apply_action_masks(self, action_space, e, obs, kg):
        (r_space, e_space), action_mask = action_space
        e_s, q, e_t, last_step, last_r, seen_nodes = obs

        # Prevent the agent from selecting the ground truth edge
        ground_truth_edge_mask = self.get_ground_truth_edge_mask(e, r_space, e_space, e_s, q, e_t, kg)
        action_mask -= ground_truth_edge_mask
        self.validate_action_mask(action_mask)

        # Mask out false negatives in the final step
        if last_step:
            false_negative_mask = self.get_false_negative_mask(e_space, e_s, q, e_t, kg)
            action_mask *= (1 - false_negative_mask)
            self.validate_action_mask(action_mask)

        # Prevent the agent from stopping in the middle of a path
        # stop_mask = (last_r == NO_OP_RELATION_ID).unsqueeze(1).float()
        # action_mask = (1 - stop_mask) * action_mask + stop_mask * (r_space == NO_OP_RELATION_ID).float()
        # Prevent loops
        # Note: avoid duplicate removal of self-loops
        # seen_nodes_b = seen_nodes[l_batch_refs]
        # loop_mask_b = (((seen_nodes_b.unsqueeze(1) == e_space.unsqueeze(2)).sum(2) > 0) *
        #      (r_space != NO_OP_RELATION_ID)).float()
        # action_mask *= (1 - loop_mask_b)
        return (r_space, e_space), action_mask

    def get_ground_truth_edge_mask(self, e, r_space, e_space, e_s, q, e_t, kg):
        ground_truth_edge_mask = \
            ((e == e_s).unsqueeze(1) * (r_space == q.unsqueeze(1)) * (e_space == e_t.unsqueeze(1)))
        inv_q = kg.get_inv_relation_id(q)
        inv_ground_truth_edge_mask = \
            ((e == e_t).unsqueeze(1) * (r_space == inv_q.unsqueeze(1)) * (e_space == e_s.unsqueeze(1)))
        return ((ground_truth_edge_mask + inv_ground_truth_edge_mask) * (e_s.unsqueeze(1) != kg.dummy_e)).float()

    def get_answer_mask(self, e_space, e_s, q, kg):
        if kg.args.mask_test_false_negatives:
            answer_vectors = kg.all_object_vectors
        else:
            answer_vectors = kg.train_object_vectors
        answer_masks = []
        for i in range(len(e_space)):
            _e_s, _q = int(e_s[i]), int(q[i])
            if not _e_s in answer_vectors or not _q in answer_vectors[_e_s]:
                answer_vector = var_cuda(torch.LongTensor([[kg.num_entities]]))
            else:
                answer_vector = answer_vectors[_e_s][_q]
            answer_mask = torch.sum(e_space[i].unsqueeze(0) == answer_vector, dim=0).long()
            answer_masks.append(answer_mask)
        answer_mask = torch.cat(answer_masks).view(len(e_space), -1)
        return answer_mask

    def get_false_negative_mask(self, e_space, e_s, q, e_t, kg):
        answer_mask = self.get_answer_mask(e_space, e_s, q, kg)
        # This is a trick applied during training where we convert a multi-answer predction problem into several
        # single-answer prediction problems. By masking out the other answers in the training set, we are forcing
        # the agent to walk towards a particular answer.
        # This trick does not affect inference on the test set: at inference time the ground truth answer will not 
        # appear in the answer mask. This can be checked by uncommenting the following assertion statement. 
        # Note that the assertion statement can trigger in the last batch if you're using a batch_size > 1 since
        # we append dummy examples to the last batch to make it the required batch size.
        # The assertion statement will also trigger in the dev set inference of NELL-995 since we randomly 
        # sampled the dev set from the training data.
        # assert(float((answer_mask * (e_space == e_t.unsqueeze(1)).long()).sum()) == 0)
        false_negative_mask = (answer_mask * (e_space != e_t.unsqueeze(1)).long()).float()
        return false_negative_mask

    def validate_action_mask(self, action_mask):
        action_mask_min = action_mask.min()
        action_mask_max = action_mask.max()
        assert (action_mask_min == 0 or action_mask_min == 1)
        assert (action_mask_max == 0 or action_mask_max == 1)

    def get_action_embedding(self, action, kg):
        """
        Return (batch) action embedding which is the concatenation of the embeddings of
        the traversed edge and the target node.

        :param action (r, e):
            (Variable:batch) indices of the most recent action
                - r is the most recently traversed edge
                - e is the destination entity.
        :param kg: Knowledge graph enviroment.
        """
        r, e = action
        relation_embedding = kg.get_relation_embeddings(r)
        if self.relation_only:
            action_embedding = relation_embedding
        else:
            entity_embedding = kg.get_entity_embeddings(e)
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
        return action_embedding

    def get_action_embedding_abs(self, action, kg):
        """
        Return (batch) action embedding which is the concatenation of the embeddings of
        the traversed edge and the target node.

        :param action (r, e):
            (Variable:batch) indices of the most recent action
                - r is the most recently traversed edge
                - e is the destination entity.
        :param kg: Knowledge graph enviroment.
        """
        r, e = action
        relation_embedding = kg.get_relation_abs_embeddings(r)
        if self.relation_only:
            action_embedding = relation_embedding
        else:
            entity_embedding = kg.get_entity_abs_embeddings(e)
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
        return action_embedding

    def define_modules(self):
        if self.relation_only:
            input_dim = self.history_dim + self.relation_dim
        elif self.relation_only_in_path:
            input_dim = self.history_dim + self.entity_dim * 2 + self.relation_dim
        else:
            input_dim = self.history_dim + self.entity_dim + self.relation_dim
        self.W1 = nn.Linear(input_dim, self.action_dim)
        self.W2 = nn.Linear(self.action_dim, self.action_dim)
        self.W1Dropout = nn.Dropout(p=self.ff_dropout_rate)
        self.W2Dropout = nn.Dropout(p=self.ff_dropout_rate)
        if self.relation_only_in_path:
            self.path_encoder = nn.LSTM(input_size=self.relation_dim,
                                        hidden_size=self.history_dim,
                                        num_layers=self.history_num_layers,
                                        batch_first=True)
        else:
            self.path_encoder = nn.LSTM(input_size=self.action_dim,
                                        hidden_size=self.history_dim,
                                        num_layers=self.history_num_layers,
                                        batch_first=True)

        if self.use_abstract_graph:
            print("pn define_abstract_modules ")
            self.define_abstract_modules()

    def define_abstract_modules(self):
        if self.relation_only:
            input_dim = self.history_dim + self.relation_dim
        elif self.relation_only_in_path:
            input_dim = self.history_dim + self.entity_dim * 2 + self.relation_dim
        else:
            input_dim = self.history_dim + self.entity_dim + self.relation_dim
        print("define_abstract_modules")
        self.W1_abstract = nn.Linear(input_dim, self.action_dim)
        self.W2_abstract = nn.Linear(self.action_dim, self.action_dim)
        self.W1Dropout_abstract = nn.Dropout(p=self.ff_dropout_rate)
        self.W2Dropout_abstract = nn.Dropout(p=self.ff_dropout_rate)
        if self.relation_only_in_path:
            self.path_abs_encoder = nn.LSTM(input_size=self.relation_dim,
                                            hidden_size=self.history_dim,
                                            num_layers=self.history_num_layers,
                                            batch_first=True)
        else:
            self.path_abs_encoder = nn.LSTM(input_size=self.action_dim,
                                            hidden_size=self.history_dim,
                                            num_layers=self.history_num_layers,
                                            batch_first=True)

    def initialize_modules(self):
        if self.xavier_initialization:
            nn.init.xavier_uniform_(self.W1.weight)
            nn.init.xavier_uniform_(self.W2.weight)
            for name, param in self.path_encoder.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

        if self.use_abstract_graph:
            self.initialize_abstract_modules()

    def initialize_abstract_modules(self):
        print("initialize_abstract_modules")
        if self.xavier_initialization:
            nn.init.xavier_uniform_(self.W1_abstract.weight)
            nn.init.xavier_uniform_(self.W2_abstract.weight)
            for name, param in self.path_abs_encoder.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
