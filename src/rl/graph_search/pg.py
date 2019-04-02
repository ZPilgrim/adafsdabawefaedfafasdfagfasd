# -*- coding: utf-8 -*-
"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Policy gradient (REINFORCE algorithm) training and inference.
"""

import torch

from src.learn_framework import LFramework
import src.rl.graph_search.beam_search as search
import src.utils.ops as ops
from src.utils.ops import int_fill_var_cuda, var_cuda, zeros_var_cuda

CHECK_NEXT_E = True


class PolicyGradient(LFramework):
    def __init__(self, args, kg, pn):
        super(PolicyGradient, self).__init__(args, kg, pn)

        # Training hyperparameters
        self.relation_only = args.relation_only
        self.save_paths_to_csv = args.save_paths_to_csv
        self.use_action_space_bucketing = args.use_action_space_bucketing
        self.num_rollouts = args.num_rollouts
        self.num_rollout_steps = args.num_rollout_steps
        self.baseline = args.baseline
        self.beta = args.beta  # entropy regularization parameter
        self.gamma = args.gamma  # shrinking factor
        self.action_dropout_rate = args.action_dropout_rate
        self.action_dropout_anneal_factor = args.action_dropout_anneal_factor
        self.action_dropout_anneal_interval = args.action_dropout_anneal_interval
        self.return_merge_scores = args.return_merge_scores
        print("DEBUG PolicyGradient return_merge_scores:", self.return_merge_scores, type(self.return_merge_scores))

        # Inference hyperparameters
        self.beam_size = args.beam_size

        # Analysis
        self.path_types = dict()
        self.num_path_types = 0

        self.path_types_abs = dict()
        self.num_path_types_abs = 0

    def reward_fun(self, e1, r, e2, pred_e2):
        return (pred_e2 == e2).float()

    def loss(self, mini_batch):
        def stablize_reward(r):
            r_2D = r.view(-1, self.num_rollouts)
            if self.baseline == 'avg_reward':
                stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
            elif self.baseline == 'avg_reward_normalized':
                stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (r_2D.std(dim=1, keepdim=True) + ops.EPSILON)
            else:
                raise ValueError('Unrecognized baseline function: {}'.format(self.baseline))
            stabled_r = stabled_r_2D.view(-1)
            return stabled_r

        e1, e2, r = self.format_batch(mini_batch, num_tiles=self.num_rollouts)
        # e1, e2, r, e1_abs, e2_abs, r_abs = self.format_batch_with_abs(mini_batch, num_tiles=self.num_rollouts)
        output = self.rollout(e1, r, e2, num_steps=self.num_rollout_steps)

        # Compute policy gradient loss
        pred_e2 = output['pred_e2']
        log_action_probs = output['log_action_probs']
        action_entropy = output['action_entropy']

        # Compute discounted reward
        final_reward = self.reward_fun(e1, r, e2, pred_e2)
        if self.baseline != 'n/a':
            final_reward = stablize_reward(final_reward)
        cum_discounted_rewards = [0] * self.num_rollout_steps
        cum_discounted_rewards[-1] = final_reward
        R = 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R = self.gamma * R + cum_discounted_rewards[i]
            cum_discounted_rewards[i] = R

        # Compute policy gradient
        pg_loss, pt_loss = 0, 0
        for i in range(self.num_rollout_steps):
            log_action_prob = log_action_probs[i]
            pg_loss += -cum_discounted_rewards[i] * log_action_prob
            pt_loss += -cum_discounted_rewards[i] * torch.exp(log_action_prob)

        # Entropy regularization
        entropy = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
        pg_loss = (pg_loss - entropy * self.beta).mean()
        pt_loss = (pt_loss - entropy * self.beta).mean()

        loss_dict = {}
        loss_dict['model_loss'] = pg_loss
        loss_dict['print_loss'] = float(pt_loss)
        loss_dict['reward'] = final_reward
        loss_dict['entropy'] = float(entropy.mean())
        if self.run_analysis:
            fn = torch.zeros(final_reward.size())
            for i in range(len(final_reward)):
                if not final_reward[i]:
                    if int(pred_e2[i]) in self.kg.all_objects[int(e1[i])][int(r[i])]:
                        fn[i] = 1
            loss_dict['fn'] = fn

        return loss_dict

    def rollout(self, e_s, q, e_t, num_steps, visualize_action_probs=False):
        """
        Perform multi-step rollout from the source entity conditioned on the query relation.
        :param pn: Policy network.
        :param e_s: (Variable:batch) source entity indices.
        :param q: (Variable:batch) query relation indices.
        :param e_t: (Variable:batch) target entity indices.
        :param kg: Knowledge graph environment.
        :param num_steps: Number of rollout steps.
        :param visualize_action_probs: If set, save action probabilities for visualization.
        :return pred_e2: Target entities reached at the end of rollout.
        :return log_path_prob: Log probability of the sampled path.
        :return action_entropy: Entropy regularization term.
        """
        assert (num_steps > 0)
        kg, pn = self.kg, self.mdl

        # Initialization
        log_action_probs = []
        action_entropy = []
        r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)
        seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)
        path_components = []

        path_trace = [(r_s, e_s)]
        pn.initialize_path((r_s, e_s), kg)

        for t in range(num_steps):
            last_r, e = path_trace[-1]
            obs = [e_s, q, e_t, t == (num_steps - 1), last_r, seen_nodes]
            db_outcomes, inv_offset, policy_entropy = pn.transit(
                e, obs, kg, use_action_space_bucketing=self.use_action_space_bucketing)
            sample_outcome = self.sample_action(db_outcomes, inv_offset)
            action = sample_outcome['action_sample']
            pn.update_path(action, kg)  # 这里的Action就是 e, r
            action_prob = sample_outcome['action_prob']
            log_action_probs.append(ops.safe_log(action_prob))
            action_entropy.append(policy_entropy)
            seen_nodes = torch.cat([seen_nodes, e.unsqueeze(1)], dim=1)
            path_trace.append(action)

            if visualize_action_probs:
                top_k_action = sample_outcome['top_actions']
                top_k_action_prob = sample_outcome['top_action_probs']
                path_components.append((e, top_k_action, top_k_action_prob))

        pred_e2 = path_trace[-1][1]
        self.record_path_trace(path_trace)

        return {
            'pred_e2': pred_e2,
            'log_action_probs': log_action_probs,
            'action_entropy': action_entropy,
            'path_trace': path_trace,
            'path_components': path_components
        }

    def rollout_with_abs(self, e_s, q, e_t, e_s_abs, q_abs, e_t_abs, num_steps, visualize_action_probs=False):
        """
        Perform multi-step rollout from the source entity conditioned on the query relation.
        :param pn: Policy network.
        :param e_s: (Variable:batch) source entity indices.
        :param q: (Variable:batch) query relation indices.
        :param e_t: (Variable:batch) target entity indices.
        :param kg: Knowledge graph environment.
        :param num_steps: Number of rollout steps.
        :param visualize_action_probs: If set, save action probabilities for visualization.
        :return pred_e2: Target entities reached at the end of rollout.
        :return log_path_prob: Log probability of the sampled path.
        :return action_entropy: Entropy regularization term.
        """
        assert (num_steps > 0)
        kg, pn = self.kg, self.mdl

        # Initialization
        log_action_probs = []
        log_action_probs_abs = []
        action_entropy = []
        action_entropy_abs = []
        r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)
        r_s_abs = int_fill_var_cuda(e_s_abs.size(), kg.dummy_start_r)
        seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)
        seen_nodes_abs = int_fill_var_cuda(e_s_abs.size(), kg.dummy_e).unsqueeze(1)
        path_components = []
        path_components_abs = []

        path_trace = [(r_s, e_s)]
        path_trace_abs = [(r_s_abs, e_s_abs)]
        pn.initialize_path((r_s, e_s), kg)
        pn.initialize_abs_path((r_s_abs, e_s_abs), kg)

        for t in range(num_steps):
            last_r, e = path_trace[-1]
            last_r_abs, e_abs = path_trace_abs[-1]
            obs = [e_s, q, e_t, t == (num_steps - 1), last_r, seen_nodes]
            obs_abs = [e_s_abs, q_abs, e_t_abs, t == (num_steps - 1), last_r_abs, seen_nodes_abs]
            db_outcomes, inv_offset, policy_entropy, db_outcomes_abs, inv_offset_abs, policy_entropy_abs = pn.transit_with_abs(
                e, obs, e_abs, obs_abs, kg, use_action_space_bucketing=self.use_action_space_bucketing)

            sample_outcome, sample_outcome_abs = self.sample_action_with_abs(e, db_outcomes, db_outcomes_abs, inv_offset,
                                                                             inv_offset_abs)
            action = sample_outcome['action_sample']
            action_abs = sample_outcome_abs['action_sample']
            pn.update_path(action, kg)  # 这里的Action就是 e, r
            pn.update_path_abs(action_abs, kg)  # 这里的Action就是 e, r
            action_prob = sample_outcome['action_prob']
            action_prob_abs = sample_outcome_abs['action_prob']
            log_action_probs.append(ops.safe_log(action_prob))
            log_action_probs_abs.append(ops.safe_log(action_prob_abs))
            action_entropy.append(policy_entropy)
            action_entropy_abs.append(policy_entropy_abs)
            seen_nodes = torch.cat([seen_nodes, e.unsqueeze(1)], dim=1)
            seen_nodes_abs = torch.cat([seen_nodes_abs, e.unsqueeze(1)], dim=1)
            path_trace.append(action)
            path_trace_abs.append(action_abs)

            if visualize_action_probs:
                top_k_action = sample_outcome['top_actions']
                top_k_action_abs = sample_outcome_abs['top_actions']
                top_k_action_prob = sample_outcome['top_action_probs']
                top_k_action_prob_abs = sample_outcome_abs['top_action_probs']
                path_components.append((e, top_k_action, top_k_action_prob))
                path_components_abs.append((e_abs, top_k_action_abs, top_k_action_prob_abs))

        pred_e2 = path_trace[-1][1]
        pred_e2_abs = path_trace_abs[-1][1]
        self.record_path_trace(path_trace)
        self.record_path_trace_abs(path_trace_abs)

        return {
                   'pred_e2': pred_e2,
                   'log_action_probs': log_action_probs,
                   'action_entropy': action_entropy,
                   'path_trace': path_trace,
                   'path_components': path_components
               }, {
                   'pred_e2': pred_e2_abs,
                   'log_action_probs': log_action_probs_abs,
                   'action_entropy': action_entropy_abs,
                   'path_trace': path_trace_abs,
                   'path_components': path_components_abs
               }

    def loss_with_abs(self, mini_batch):

        def stablize_reward(r):
            r_2D = r.view(-1, self.num_rollouts)
            if self.baseline == 'avg_reward':
                stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
            elif self.baseline == 'avg_reward_normalized':
                stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (r_2D.std(dim=1, keepdim=True) + ops.EPSILON)
            else:
                raise ValueError('Unrecognized baseline function: {}'.format(self.baseline))
            stabled_r = stabled_r_2D.view(-1)
            return stabled_r

        e1, e2, r, e1_abs, e2_abs, r_abs = self.format_batch_with_abs(mini_batch, num_tiles=self.num_rollouts)
        output, output_abs = self.rollout_with_abs(e1, r, e2, e1_abs, r_abs, e2_abs, num_steps=self.num_rollout_steps)

        # Compute policy gradient loss
        pred_e2 = output['pred_e2']
        pred_e2_abs = output_abs['pred_e2']
        log_action_probs = output['log_action_probs']

        log_action_probs_abs = output_abs['log_action_probs']
        action_entropy = output['action_entropy']
        action_entropy_abs = output_abs['action_entropy']

        # Compute discounted reward
        final_reward = self.reward_fun(e1, r, e2, pred_e2)
        final_reward_abs = final_reward  # self.reward_fun(e1_abs, r_abs, e2_abs, pred_e2_abs)
        # final_reward_abs = self.reward_fun(e1_abs, r_abs, e2_abs, pred_e2_abs)
        if self.baseline != 'n/a':
            final_reward = stablize_reward(final_reward)
            final_reward_abs = stablize_reward(final_reward_abs)
        cum_discounted_rewards = [0] * self.num_rollout_steps
        cum_discounted_rewards_abs = [0] * self.num_rollout_steps
        cum_discounted_rewards[-1] = final_reward
        cum_discounted_rewards_abs[-1] = final_reward_abs
        R = 0
        R_abs = 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R = self.gamma * R + cum_discounted_rewards[i]
            R_abs = self.gamma * R_abs + cum_discounted_rewards_abs[i]
            cum_discounted_rewards[i] = R
            cum_discounted_rewards_abs[i] = R_abs

        # Compute policy gradient
        pg_loss, pt_loss = 0, 0
        pg_loss_abs, pt_loss_abs = 0, 0
        for i in range(self.num_rollout_steps):
            log_action_prob = log_action_probs[i]
            log_action_prob_abs = log_action_probs_abs[i]
            pg_loss += -cum_discounted_rewards[i] * log_action_prob
            pg_loss_abs += -cum_discounted_rewards_abs[i] * log_action_prob_abs
            pt_loss += -cum_discounted_rewards[i] * torch.exp(log_action_prob)
            pt_loss_abs += -cum_discounted_rewards_abs[i] * torch.exp(log_action_prob_abs)

        # Entropy regularization
        entropy = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
        entropy_abs = torch.cat([x.unsqueeze(1) for x in action_entropy_abs], dim=1).mean(dim=1)
        pg_loss = (pg_loss - entropy * self.beta).mean()
        pg_loss_abs = (pg_loss_abs - entropy_abs * self.beta).mean()
        pt_loss = (pt_loss - entropy * self.beta).mean()
        pt_loss_abs = (pt_loss_abs - entropy_abs * self.beta).mean()

        def print_grad(grad):
            print(grad)

        loss_dict = {}
        loss_dict_abs = {}
        loss_dict['model_loss'] = pg_loss
        # pg_loss.register_hook(print_grad)
        loss_dict_abs['model_loss'] = pg_loss_abs
        # print("=======")
        # pg_loss_abs.register_hook(print_grad)
        loss_dict['print_loss'] = float(pt_loss)
        loss_dict_abs['print_loss'] = float(pt_loss_abs)
        loss_dict['reward'] = final_reward
        loss_dict_abs['reward'] = final_reward_abs
        loss_dict['entropy'] = float(entropy.mean())
        loss_dict_abs['entropy'] = float(entropy_abs.mean())
        if self.run_analysis:
            fn = torch.zeros(final_reward.size())
            for i in range(len(final_reward)):
                if not final_reward[i]:
                    if int(pred_e2[i]) in self.kg.all_objects[int(e1[i])][int(r[i])]:
                        fn[i] = 1
            loss_dict['fn'] = fn

            fn_abs = torch.zeros(final_reward_abs.size())
            for i in range(len(final_reward_abs)):
                if not final_reward_abs[i]:
                    if int(pred_e2_abs[i]) in self.kg.all_objects_abs[int(e1_abs[i])][int(r_abs[i])]:
                        fn_abs[i] = 1
            loss_dict_abs['fn'] = fn_abs

        return loss_dict, loss_dict_abs

    def sample_action(self, db_outcomes, inv_offset=None):
        """
        Sample an action based on current policy.
        :param db_outcomes (((r_space, e_space), action_mask), action_dist):
                r_space: (Variable:batch) relation space
                e_space: (Variable:batch) target entity space
                action_mask: (Variable:batch) binary mask indicating padding actions.
                action_dist: (Variable:batch) action distribution of the current step based on set_policy
                    network parameters
        :param inv_offset: Indexes for restoring original order in a batch.
        :return next_action (next_r, next_e): Sampled next action.
        :return action_prob: Probability of the sampled action.
        """

        def apply_action_dropout_mask(action_dist, action_mask):
            if self.action_dropout_rate > 0:
                rand = torch.rand(action_dist.size())
                action_keep_mask = var_cuda(rand > self.action_dropout_rate).float()
                # There is a small chance that that action_keep_mask is accidentally set to zero.
                # When this happen, we take a random sample from the available actions.
                # sample_action_dist = action_dist * (action_keep_mask + ops.EPSILON)
                sample_action_dist = \
                    action_dist * action_keep_mask + ops.EPSILON * (1 - action_keep_mask) * action_mask
                return sample_action_dist
            else:
                return action_dist

        def sample(action_space, action_dist):
            sample_outcome = {}
            ((r_space, e_space), action_mask) = action_space
            sample_action_dist = apply_action_dropout_mask(action_dist, action_mask)
            idx = torch.multinomial(sample_action_dist, 1, replacement=True)
            next_r = ops.batch_lookup(r_space, idx)
            next_e = ops.batch_lookup(e_space, idx)
            action_prob = ops.batch_lookup(action_dist, idx)
            sample_outcome['action_sample'] = (next_r, next_e)
            sample_outcome['action_prob'] = action_prob
            return sample_outcome

        if inv_offset is not None:
            next_r_list = []
            next_e_list = []
            action_dist_list = []
            action_prob_list = []
            for action_space, action_dist in db_outcomes:
                sample_outcome = sample(action_space, action_dist)
                next_r_list.append(sample_outcome['action_sample'][0])
                next_e_list.append(sample_outcome['action_sample'][1])
                action_prob_list.append(sample_outcome['action_prob'])
                action_dist_list.append(action_dist)
            next_r = torch.cat(next_r_list, dim=0)[inv_offset]
            next_e = torch.cat(next_e_list, dim=0)[inv_offset]
            action_sample = (next_r, next_e)
            action_prob = torch.cat(action_prob_list, dim=0)[inv_offset]
            sample_outcome = {}
            sample_outcome['action_sample'] = action_sample
            sample_outcome['action_prob'] = action_prob
        else:
            sample_outcome = sample(db_outcomes[0][0], db_outcomes[0][1])

        return sample_outcome

    def sample_action_with_abs(self, last_e, db_outcomes, db_outcomes_abs, inv_offset=None, inv_offset_abs=None):
        """
        Sample an action based on current policy.
        :param db_outcomes (((r_space, e_space), action_mask), action_dist):
                r_space: (Variable:batch) relation space
                e_space: (Variable:batch) target entity space
                action_mask: (Variable:batch) binary mask indicating padding actions.
                action_dist: (Variable:batch) action distribution of the current step based on set_policy
                    network parameters
        :param inv_offset: Indexes for restoring original order in a batch.
        :return next_action (next_r, next_e): Sampled next action.
        :return action_prob: Probability of the sampled action.
        """

        global CHECK_NEXT_E

        def apply_action_dropout_mask(action_dist, action_mask):
            if self.action_dropout_rate > 0:
                rand = torch.rand(action_dist.size())
                action_keep_mask = var_cuda(rand > self.action_dropout_rate).float()
                # There is a small chance that that action_keep_mask is accidentally set to zero.
                # When this happen, we take a random sample from the available actions.
                # sample_action_dist = action_dist * (action_keep_mask + ops.EPSILON)
                sample_action_dist = \
                    action_dist * action_keep_mask + ops.EPSILON * (1 - action_keep_mask) * action_mask
                return sample_action_dist
            else:
                return action_dist

        def sample(action_space, action_dist):
            sample_outcome = {}
            ((r_space, e_space), action_mask) = action_space
            sample_action_dist = apply_action_dropout_mask(action_dist, action_mask)
            idx = torch.multinomial(sample_action_dist, 1, replacement=True)
            next_r = ops.batch_lookup(r_space, idx)
            next_e = ops.batch_lookup(e_space, idx)
            action_prob = ops.batch_lookup(action_dist, idx)
            sample_outcome['action_sample'] = (next_r, next_e)
            sample_outcome['action_prob'] = action_prob
            return sample_outcome


        def es2ts(es):
            ret = []
            for e in es:
                ret.append(self.kg.get_typeid(e))
            #print("ret ES===>>", len(set(ret)))
            ret = var_cuda(torch.LongTensor(ret), requires_grad=False)
            return ret

        def sample_with_abs(action_space, action_dist, action_space_abs, action_dist_abs):
            sample_outcome = {}
            sample_outcome_abs = {}
            ((r_space, e_space), action_mask) = action_space
            ((r_space_abs, e_space_abs), action_mask_abs) = action_space_abs

            sample_action_dist = apply_action_dropout_mask(action_dist, action_mask)
            idx = torch.multinomial(sample_action_dist, 1, replacement=True)
            next_r = ops.batch_lookup(r_space, idx)
            next_e = ops.batch_lookup(e_space, idx)

            next_r_abs = next_r
            next_e_abs = es2ts(next_e)

            # tot_set = set()
            # tot_set_er = set()
            # tot_set_abs_er = set()
            # for e, r in zip(next_r_abs.tolist(), next_e_abs.tolist()):
            #     tot_set.add((e,r))
            # for _e, _r in zip(next_r.tolist(), next_e.tolist()):
            #     tot_set_abs_er.add((_r, self.kg.get_typeid(_e)))
            #     tot_set_er.add((_r,_e))
            # print("SET ES===>>", len(tot_set), "e_space_abs.size():", e_space_abs.size(), "r_space_abs.size():",r_space_abs.size(), "next_r_abs:", next_r_abs.size(), "next_e_abs:", next_e_abs.size(), "e_space.size():", e_space.size(), "tot_set_abs_er:", len(tot_set_abs_er), "tot_set_er:", len(tot_set_er))

            type_mask = (next_e_abs.view(-1, 1) == e_space_abs)
            r_mask = (next_r_abs.view(-1, 1) == r_space_abs)
            action_mask_abs = r_mask.mul(type_mask)
            if (action_mask_abs == 1).nonzero().size()[0] != next_e_abs.size()[0]:
                print(action_mask_abs.size())
                print(r_mask.size())
                print(type_mask.size())
                print(
                    "----------------------------GETERROR-------------------------------")
                for _ in range(r_space.size()[0]):
                    if torch.sum(action_mask_abs[_, :]) == 0:
                        # print("r_space_abs")
                        # print(r_space_abs[_, :])
                        # print("e_space_abs")
                        # print(e_space_abs[_, :])
                        # print("next_e_abs")
                        # print(next_e_abs[_])
                        # print("next_r_abs")
                        # print(next_r_abs[_]
                        for i in range(len(e_space_abs[_, :])):
                            print("{},{}==>{},{}; search for r,e:({},{}), abs_r,abs_e:({},{}); r_mask:{}; type_mask:{}; action_mask:{}; e_s:{}".format(
                                r_space[_, i], e_space[_, i], r_space_abs[_, i], e_space_abs[_, i], next_r[_], next_e[_], next_r_abs[_], 
                                next_e_abs[_], r_mask[_, i], type_mask[_, i], action_mask_abs[_, i], last_e[_]))

                        assert(1 == 0)
        
            action_prob_abs = torch.masked_select(action_dist_abs, action_mask_abs)


            action_prob = ops.batch_lookup(action_dist, idx)
            #print ("===> action_prob_abs:", action_prob_abs.size(), " action_prob:", action_prob.size())
            sample_outcome['action_sample'] = (next_r, next_e)
            sample_outcome_abs['action_sample'] = (next_r_abs, next_e_abs)
            sample_outcome['action_prob'] = action_prob
            sample_outcome_abs['action_prob'] = action_prob_abs
            return sample_outcome, sample_outcome_abs

        if inv_offset is not None:
            next_r_list = []
            next_r_list_abs = []
            next_e_list = []
            next_e_list_abs = []
            action_dist_list = []
            action_dist_list_abs = []
            action_prob_list = []
            action_prob_list_abs = []
            # TODO:CHECK这里... 尤其上面的sample apply_action_dropout_mask, 这里概率部分是不是有这样的可能，两条path到达同样的抽象实体 所以概率不能合并。。。 主要概率是否影响更新...
            for i, (action_space, action_dist) in enumerate(db_outcomes):
                action_space_abs = db_outcomes_abs[i][0]
                action_dist_abs = db_outcomes_abs[i][1]

                sample_outcome, sample_outcome_abs = sample_with_abs(action_space, action_dist, action_space_abs, action_dist_abs)
                next_r_list.append(sample_outcome['action_sample'][0])
                next_r_list_abs.append(sample_outcome_abs['action_sample'][0])

                next_e_list.append(sample_outcome['action_sample'][1])
                next_e_list_abs.append(sample_outcome_abs['action_sample'][1])
                # # if CHECK_NEXT_E:
                # #     print ("type:", type(next_e_list[-1]), next_e_list[-1])
                # #     try:
                # #         print("next_e_list[-1].tolist()", next_e_list[-1].tolist())
                # #     except Exception as e:
                # #         print("e:", e)
                # #
                # #     CHECK_NEXT_E = False
                #
                # type2prob = mk_type2prob_map(action_space_abs, action_dist_abs)
                #
                # _real_type_list = []
                # _abs_type_prob_list = []
                # for i, _next_e in enumerate(sample_outcome['action_sample'][1]):
                #     _real_type_list.append(self.kg.get_typeid(_next_e))
                #     _abs_type_prob_list.append(sample_outcome['action_prob'][i])
                #
                # for _e in next_e_list[-1]:
                #     _real_type_list.append(self.kg.get_typeid(_e))
                # _real_type_list = var_cuda(torch.LongTensor(_real_type_list),
                #                            requires_grad=False)


                # next_e_list_abs.append(self.kg.get_typeid(next_e_list[-1]))

                action_prob_list.append(sample_outcome['action_prob'])
                action_prob_list_abs.append(sample_outcome_abs['action_prob'])
                action_dist_list.append(action_dist)
                action_dist_list_abs.append(action_dist_abs)  # TODO:CHECK

            next_r_abs = torch.cat(next_r_list_abs, dim=0)[inv_offset_abs]
            next_r = torch.cat(next_r_list, dim=0)[inv_offset]
            next_e = torch.cat(next_e_list, dim=0)[inv_offset]
            next_e_abs = torch.cat(next_e_list_abs, dim=0)[inv_offset_abs]
            action_sample = (next_r, next_e)
            action_sample_abs = (next_r_abs, next_e_abs)
            action_prob = torch.cat(action_prob_list, dim=0)[inv_offset]
            action_prob_abs = torch.cat(action_prob_list_abs, dim=0)[inv_offset_abs]
            sample_outcome = {}
            sample_outcome_abs = {}
            sample_outcome['action_sample'] = action_sample
            sample_outcome_abs['action_sample'] = action_sample_abs
            sample_outcome['action_prob'] = action_prob
            sample_outcome_abs['action_prob'] = action_prob_abs
            # print("===>out ")
        else:
            # sample_outcome = sample(db_outcomes[0][0], db_outcomes[0][1])
            # sample_outcome_abs = sample(db_outcomes_abs[0][0], db_outcomes_abs[0][1])
            action_space_abs = db_outcomes_abs[0][0]
            action_dist_abs = db_outcomes_abs[0][1]

            sample_outcome, sample_outcome_abs = sample_with_abs(db_outcomes[0][0], db_outcomes[0][1], action_space_abs,
                                                                 action_dist_abs)

        return sample_outcome, sample_outcome_abs

    def predict(self, mini_batch, verbose=False):
        # return_merge_scores= None #'sum'
        # return_merge_scores= 'sum'

        # print("return_merge_scores:", self.return_merge_scores, type(self.return_merge_scores))
        kg, pn = self.kg, self.mdl
        e1, e2, r = self.format_batch(mini_batch)
        beam_search_output = search.beam_search(
            pn, e1, r, e2, kg, self.num_rollout_steps, self.beam_size, return_merge_scores=self.return_merge_scores)
        pred_e2s = beam_search_output['pred_e2s']
        pred_e2_scores = beam_search_output['pred_e2_scores']

        if verbose:
            # print inference paths
            search_traces = beam_search_output['search_traces']
            output_beam_size = min(self.beam_size, pred_e2_scores.shape[1])
            for i in range(len(e1)):
                for j in range(output_beam_size):
                    ind = i * output_beam_size + j
                    if pred_e2s[i][j] == kg.dummy_e:
                        break
                    search_trace = []
                    for k in range(len(search_traces)):
                        search_trace.append((int(search_traces[k][0][ind]), int(search_traces[k][1][ind])))
                    print('beam {}: score = {} \n<PATH> {}'.format(
                        j, float(pred_e2_scores[i][j]), ops.format_path(search_trace, kg)))

        with torch.no_grad():
            pred_scores = zeros_var_cuda([len(e1), kg.num_entities])
            for i in range(len(e1)):
                if self.return_merge_scores == 'sum' or self.return_merge_scores == 'mean':
                    pred_scores[i][pred_e2s[i].tolist()] = pred_e2_scores[i]
                else:
                    pred_scores[i][pred_e2s[i]] = torch.exp(pred_e2_scores[i])
        return pred_scores

    def predict_abs(self, mini_batch, verbose=False):
        # return_merge_scores= None #'sum'
        # return_merge_scores= 'sum'

        # print("return_merge_scores:", self.return_merge_scores, type(self.return_merge_scores))
        kg, pn = self.kg, self.mdl
        e1_abs, e2_abs, r_abs = self.format_batch(mini_batch)
        # _, _, _, e1_abs, e2_abs, r_abs = self.format_batch_with_abs(mini_batch)
        beam_search_output = search.beam_search_abs(
            pn, e1_abs, r_abs, e2_abs, kg, self.num_rollout_steps, self.beam_size,
            return_path_components=self.save_paths_to_csv,
            return_merge_scores=self.return_merge_scores)  # TODO:修改这里走beam_search_abs
        pred_e2s = beam_search_output['pred_e2s']
        pred_e2_scores = beam_search_output['pred_e2_scores']

        if verbose:
            # print inference paths
            search_traces = beam_search_output['search_traces']
            output_beam_size = min(self.beam_size, pred_e2_scores.shape[1])
            for i in range(len(e1_abs)):
                for j in range(output_beam_size):
                    ind = i * output_beam_size + j
                    if pred_e2s[i][j] == kg.dummy_e:
                        break
                    search_trace = []
                    for k in range(len(search_traces)):
                        search_trace.append((int(search_traces[k][0][ind]), int(search_traces[k][1][ind])))
                    print('beam {}: score = {} \n<PATH> {}'.format(
                        j, float(pred_e2_scores[i][j]), ops.format_path(search_trace, kg)))

        with torch.no_grad():
            pred_scores = zeros_var_cuda([len(e1_abs), kg.num_entities])
            for i in range(len(e1_abs)):
                if self.return_merge_scores == 'sum' or self.return_merge_scores == 'mean':
                    pred_scores[i][pred_e2s[i].tolist()] = pred_e2_scores[i]
                else:
                    pred_scores[i][pred_e2s[i]] = torch.exp(pred_e2_scores[i])
        return pred_scores

    def record_path_trace(self, path_trace):
        path_length = len(path_trace)
        flattened_path_trace = [x for t in path_trace for x in t]
        path_trace_mat = torch.cat(flattened_path_trace).reshape(-1, path_length)
        path_trace_mat = path_trace_mat.data.cpu().numpy()

        for i in range(path_trace_mat.shape[0]):
            path_recorder = self.path_types
            for j in range(path_trace_mat.shape[1]):
                e = path_trace_mat[i, j]
                if not e in path_recorder:
                    if j == path_trace_mat.shape[1] - 1:
                        path_recorder[e] = 1
                        self.num_path_types += 1
                    else:
                        path_recorder[e] = {}
                else:
                    if j == path_trace_mat.shape[1] - 1:
                        path_recorder[e] += 1
                path_recorder = path_recorder[e]

    def record_path_trace_abs(self, path_trace):
        path_length = len(path_trace)
        flattened_path_trace = [x for t in path_trace for x in t]
        path_trace_mat = torch.cat(flattened_path_trace).reshape(-1, path_length)
        path_trace_mat = path_trace_mat.data.cpu().numpy()

        for i in range(path_trace_mat.shape[0]):
            path_recorder = self.path_types_abs
            for j in range(path_trace_mat.shape[1]):
                e = path_trace_mat[i, j]
                if not e in path_recorder:
                    if j == path_trace_mat.shape[1] - 1:
                        path_recorder[e] = 1
                        self.num_path_types_abs += 1
                    else:
                        path_recorder[e] = {}
                else:
                    if j == path_trace_mat.shape[1] - 1:
                        path_recorder[e] += 1
                path_recorder = path_recorder[e]
