# -*- coding: utf-8 -*-
"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Base learning framework.
"""

import os
import random
import shutil
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import src.eval
from src.utils.ops import var_cuda, zeros_var_cuda
import src.utils.ops as ops


class LFramework(nn.Module):
    def __init__(self, args, kg, mdl):
        super(LFramework, self).__init__()
        self.data_dir = args.data_dir
        self.model_dir = args.model_dir
        self.model = args.model

        # Training hyperparameters
        self.use_abstract_graph = args.use_abstract_graph
        self.batch_size = args.batch_size
        self.train_batch_size = args.train_batch_size
        self.dev_batch_size = args.dev_batch_size
        self.start_epoch = args.start_epoch
        self.num_epochs = args.num_epochs
        self.num_wait_epochs = args.num_wait_epochs
        self.num_peek_epochs = args.num_peek_epochs
        self.learning_rate = args.learning_rate
        self.grad_norm = args.grad_norm
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.optim = None
        self.merge_abs_real_score = args.merge_abs_real_score
        print("LFramework:", self.merge_abs_real_score)

        self.inference = not args.train
        self.run_analysis = args.run_analysis

        self.kg = kg
        self.mdl = mdl
        print('{} module created. abstract_graph:{}'.format(self.model, self.use_abstract_graph))

    def print_all_model_parameters(self):
        print('\nModel Parameters')
        print('--------------------------')
        for name, param in self.named_parameters():
            print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))
        param_sizes = [param.numel() for param in self.parameters()]
        print('Total # parameters = {}'.format(sum(param_sizes)))
        print('--------------------------')
        print()

    def run_train(self, train_data, dev_data):
        self.print_all_model_parameters()

        if self.optim is None:
            self.optim = optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

        # Track dev metrics changes
        best_dev_metrics = 0
        dev_metrics_history = []

        for epoch_id in range(self.start_epoch, self.num_epochs):
            print('Epoch {}'.format(epoch_id))
            if self.rl_variation_tag.startswith('rs'):
                # Reward shaping module sanity check:
                #   Make sure the reward shaping module output value is in the correct range
                train_scores = self.test_fn(train_data)
                dev_scores = self.test_fn(dev_data)
                print('Train set average fact score: {}'.format(float(train_scores.mean())))
                print('Dev set average fact score: {}'.format(float(dev_scores.mean())))

            # Update model parameters
            self.train()
            if self.rl_variation_tag.startswith('rs'):
                self.fn.eval()
                self.fn_kg.eval()
                if self.model.endswith('hypere'):
                    self.fn_secondary_kg.eval()
            self.batch_size = self.train_batch_size
            random.shuffle(train_data)
            batch_losses = []
            entropies = []
            if self.run_analysis:
                rewards = None
                fns = None
            for example_id in tqdm(range(0, len(train_data), self.batch_size)):

                self.optim.zero_grad()

                mini_batch = train_data[example_id:example_id + self.batch_size]
                if len(mini_batch) < self.batch_size:
                    continue
                loss = self.loss(mini_batch)
                loss['model_loss'].backward()
                if self.grad_norm > 0:
                    clip_grad_norm_(self.parameters(), self.grad_norm)

                self.optim.step()

                batch_losses.append(loss['print_loss'])
                if 'entropy' in loss:
                    entropies.append(loss['entropy'])
                if self.run_analysis:
                    if rewards is None:
                        rewards = loss['reward']
                    else:
                        rewards = torch.cat([rewards, loss['reward']])
                    if fns is None:
                        fns = loss['fn']
                    else:
                        fns = torch.cat([fns, loss['fn']])
            # Check training statistics
            stdout_msg = 'Epoch {}: average training loss = {}'.format(epoch_id, np.mean(batch_losses))
            if entropies:
                stdout_msg += 'entropy = {}'.format(np.mean(entropies))
            print(stdout_msg)
            self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id)
            if self.run_analysis:
                print('* Analysis: # path types seen = {}'.format(self.num_path_types))
                num_hits = float(rewards.sum())
                hit_ratio = num_hits / len(rewards)
                print('* Analysis: # hits = {} ({})'.format(num_hits, hit_ratio))
                num_fns = float(fns.sum())
                fn_ratio = num_fns / len(fns)
                print('* Analysis: false negative ratio = {}'.format(fn_ratio))

            # Check dev set performance
            if self.run_analysis or (epoch_id > 0 and epoch_id % self.num_peek_epochs == 0):
                self.eval()
                self.batch_size = self.dev_batch_size
                dev_scores = self.forward(dev_data, abs_graph=False, verbose=False)
                print('Dev set performance: (correct evaluation)')
                _, _, _, _, mrr = src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.dev_objects, verbose=True)
                metrics = mrr
                print('Dev set performance: (include test set labels)')
                src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.all_objects, verbose=True)
                # Action dropout anneaking
                if self.model.startswith('point'):
                    eta = self.action_dropout_anneal_interval
                    if len(dev_metrics_history) > eta and metrics < min(dev_metrics_history[-eta:]):
                        old_action_dropout_rate = self.action_dropout_rate
                        self.action_dropout_rate *= self.action_dropout_anneal_factor
                        print('Decreasing action dropout rate: {} -> {}'.format(
                            old_action_dropout_rate, self.action_dropout_rate))
                # Save checkpoint
                if metrics > best_dev_metrics:
                    self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id, is_best=True)
                    best_dev_metrics = metrics
                    with open(os.path.join(self.model_dir, 'best_dev_iteration.dat'), 'w') as o_f:
                        o_f.write('{}'.format(epoch_id))
                else:
                    # Early stopping
                    if epoch_id >= self.num_wait_epochs and metrics < np.mean(
                            dev_metrics_history[-self.num_wait_epochs:]):
                        break
                dev_metrics_history.append(metrics)
                if self.run_analysis:
                    num_path_types_file = os.path.join(self.model_dir, 'num_path_types.dat')
                    dev_metrics_file = os.path.join(self.model_dir, 'dev_metrics.dat')
                    hit_ratio_file = os.path.join(self.model_dir, 'hit_ratio.dat')
                    fn_ratio_file = os.path.join(self.model_dir, 'fn_ratio.dat')
                    if epoch_id == 0:
                        with open(num_path_types_file, 'w') as o_f:
                            o_f.write('{}\n'.format(self.num_path_types))
                        with open(dev_metrics_file, 'w') as o_f:
                            o_f.write('{}\n'.format(metrics))
                        with open(hit_ratio_file, 'w') as o_f:
                            o_f.write('{}\n'.format(hit_ratio))
                        with open(fn_ratio_file, 'w') as o_f:
                            o_f.write('{}\n'.format(fn_ratio))
                    else:
                        with open(num_path_types_file, 'a') as o_f:
                            o_f.write('{}\n'.format(self.num_path_types))
                        with open(dev_metrics_file, 'a') as o_f:
                            o_f.write('{}\n'.format(metrics))
                        with open(hit_ratio_file, 'a') as o_f:
                            o_f.write('{}\n'.format(hit_ratio))
                        with open(fn_ratio_file, 'a') as o_f:
                            o_f.write('{}\n'.format(fn_ratio))

    def run_train_with_abstract(self, train_data, dev_data):
        self.print_all_model_parameters()

        if self.optim is None:
            self.optim = optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

        # Track dev metrics changes
        best_dev_metrics = 0
        dev_metrics_history = []

        FORCE_MERGE = True
        # print("FORCE TRAIN2 128, FORCE_MERGE")
        # train_data = train_data[:128]

        for epoch_id in range(self.start_epoch, self.num_epochs):
            print('Epoch {}'.format(epoch_id))
            if self.rl_variation_tag.startswith('rs'):
                # TODO:CHECK 这里的修改
                # Reward shaping module sanity check:
                #   Make sure the reward shaping module output value is in the correct range
                train_scores = self.test_fn(train_data)
                dev_scores = self.test_fn(dev_data)
                print('Train set average fact score: {}'.format(float(train_scores.mean())))
                print('Dev set average fact score: {}'.format(float(dev_scores.mean())))

            # Update model parameters
            self.train()
            if self.rl_variation_tag.startswith('rs'):
                self.fn.eval()
                self.fn_kg.eval()
                if self.model.endswith('hypere'):
                    self.fn_secondary_kg.eval()
            self.batch_size = self.train_batch_size
            random.shuffle(train_data)
            batch_losses = []
            batch_losses_abs = []
            entropies = []
            entropies_abs = []
            if self.run_analysis:
                rewards = None
                rewards_abs = None
                fns = None
                fns_abs = None
            for example_id in tqdm(range(0, len(train_data), self.batch_size)):

                self.optim.zero_grad()

                mini_batch = train_data[example_id:example_id + self.batch_size]
                if len(mini_batch) < self.batch_size:
                    continue
                loss, loss_abs = self.loss_with_abs(mini_batch)

                # loss['model_loss'].backward(retain_graph=True)


                def force_set_grad(requires_grad):
                    names = [
                        'kg.entity_embeddings.weight',

                    ]
                    self.kg.entity_embeddings.weight.requires_grad = requires_grad

                # self.kg
                # print ("params:", self.parameters())
                # self.named_parameters()

                force_set_grad(True)
                loss['model_loss'].backward()
                # loss_abs['model_loss'].backward()
                if self.grad_norm > 0:
                    clip_grad_norm_(self.parameters(), self.grad_norm)

                self.optim.step()
                force_set_grad(False)
                # print("DEBUG CHECK PARAMS...")
                # for name, params in self.named_parameters(recurse=True):
                #     print("name:", name, "grad:", params.requires_grad)

                loss_abs['model_loss'].backward()
                if self.grad_norm > 0:
                    clip_grad_norm_(self.parameters(), self.grad_norm)

                self.optim.step()

                batch_losses.append(loss['print_loss'])
                batch_losses_abs.append(loss_abs['print_loss'])
                if 'entropy' in loss:
                    entropies.append(loss['entropy'])
                    entropies_abs.append(loss_abs['entropy'])
                if self.run_analysis:
                    if rewards is None:
                        rewards = loss['reward']
                        rewards_abs = loss_abs['reward']
                    else:
                        rewards = torch.cat([rewards, loss['reward']])
                        rewards_abs = torch.cat([rewards_abs, loss_abs['reward']])
                    if fns is None:
                        fns = loss['fn']
                        fns_abs = loss_abs['fn']
                    else:
                        fns = torch.cat([fns, loss['fn']])
                        fns_abs = torch.cat([fns_abs, loss_abs['fn']])
            # Check training statistics
            stdout_msg = 'Epoch {}: average training loss = {} loss_abs={} '.format(epoch_id, np.mean(batch_losses),
                                                                                    np.mean(batch_losses_abs))
            if entropies:
                stdout_msg += 'entropy = {}'.format(np.mean(entropies))
                stdout_msg += ' entropy_abs = {}'.format(np.mean(entropies_abs))
            print(stdout_msg)
            self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id)
            if self.run_analysis:
                print('* Analysis: # path types seen = {}'.format(self.num_path_types))
                num_hits = float(rewards.sum())
                num_hits_abs = float(rewards_abs.sum())
                hit_ratio = num_hits / len(rewards)
                hit_ratio_abs = num_hits_abs / len(rewards)
                print('* Analysis: # hits = {} ({})'.format(num_hits, hit_ratio))
                print('* Analysis: # hits_abs = {} ({})'.format(num_hits_abs, hit_ratio_abs))
                num_fns = float(fns.sum())
                num_fns_abs = float(fns_abs.sum())
                fn_ratio = num_fns / len(fns)
                fn_ratio_abs = num_fns_abs / len(fns_abs)
                print('* Analysis: false negative ratio = {} {}'.format(fn_ratio, fn_ratio_abs))

            # Check dev set performance
            if self.run_analysis or (epoch_id > 0 and epoch_id % self.num_peek_epochs == 0):
                self.eval()
                self.batch_size = self.dev_batch_size

                dev_scores = self.forward(dev_data, verbose=False)
                print('Dev set performance: (correct evaluation)')
                _, _, _, _, mrr = src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.dev_objects, verbose=True)
                metrics = mrr
                print('Dev set performance: (include test set labels)')
                src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.all_objects, verbose=True)

                # dev_scores_abs = self.forward(dev_data, verbose=False, abs_graph=True)
                # print('Dev set performance of abs model on abs graph: (correct evaluation)')
                # _, _, _, _, mrr = src.eval.hits_and_ranks(
                #     dev_data, dev_scores_abs, self.kg.dev_objects_abs, verbose=True)
                # metrics = mrr
                # print(
                #     'Dev set performance of abs model on abs graph: (correct evaluation: (include test set labels)')
                # src.eval.hits_and_ranks(
                #     dev_data, dev_scores, self.kg.all_objects_abs, verbose=True)

                dev_scores_abs2real = self.forward(dev_data, verbose=False, same_infer=True)
                print(
                    'Dev set performance of abs model on ori graph: (correct evaluation)')
                _, _, _, _, mrr = src.eval.hits_and_ranks(
                    dev_data, dev_scores_abs2real, self.kg.dev_objects, verbose=True)
                metrics = mrr
                print(
                    'Dev set performance of abs model on ori graph: (include test set labels)')
                src.eval.hits_and_ranks(
                    dev_data, dev_scores_abs2real, self.kg.all_objects, verbose=True)

                # merge
                dev_scores_force_merge = self.merge_abs_real_score * dev_scores + (
                                                                                      1.0 - self.merge_abs_real_score) * dev_scores_abs2real

                _, _, _, _, mrr = src.eval.hits_and_ranks(
                    dev_data, dev_scores_force_merge, self.kg.dev_objects, verbose=True)
                metrics = mrr
                print(
                    'Dev set performance of abs model force_merge: (include test set labels)')
                src.eval.hits_and_ranks(
                    dev_data, dev_scores_force_merge, self.kg.all_objects, verbose=True)

                # from src.rl.graph_search.beam_search import REAL_ALL_PATHS, SAME_ALL_PATHS
                #merge by path， 如果real_path和abs_real_path的type一样 就认为可以merge


                _, _, _, _, mrr = src.eval.hits_and_ranks_merge(
                    dev_data, dev_scores, self.kg.all_objects, dev_scores_abs2real, self.merge_abs_real_score, self.kg.entity2typeid, verbose=True)
                metrics = mrr
                print(
                    'Dev set performance of abs model force_merge same type: (include test set labels)')
                src.eval.hits_and_ranks_merge(
                    dev_data, dev_scores, self.kg.all_objects, dev_scores_abs2real, self.merge_abs_real_score, self.kg.entity2typeid, verbose=True)


                # global REAL_ALL_PATHS, SAME_ALL_PATHS
                # REAL_ALL_PATHS, SAME_ALL_PATHS = [], []

                # Action dropout anneaking
                if self.model.startswith('point'):
                    eta = self.action_dropout_anneal_interval
                    if len(dev_metrics_history) > eta and metrics < min(dev_metrics_history[-eta:]):
                        old_action_dropout_rate = self.action_dropout_rate
                        self.action_dropout_rate *= self.action_dropout_anneal_factor
                        print('Decreasing action dropout rate: {} -> {}'.format(
                            old_action_dropout_rate, self.action_dropout_rate))
                # Save checkpoint
                if metrics > best_dev_metrics:
                    self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id, is_best=True)
                    best_dev_metrics = metrics
                    with open(os.path.join(self.model_dir, 'best_dev_iteration.dat'), 'w') as o_f:
                        o_f.write('{}'.format(epoch_id))
                else:
                    # Early stopping
                    if epoch_id >= self.num_wait_epochs and metrics < np.mean(
                            dev_metrics_history[-self.num_wait_epochs:]):
                        break
                dev_metrics_history.append(metrics)
                if self.run_analysis:
                    num_path_types_file = os.path.join(self.model_dir, 'num_path_types.dat')
                    dev_metrics_file = os.path.join(self.model_dir, 'dev_metrics.dat')
                    hit_ratio_file = os.path.join(self.model_dir, 'hit_ratio.dat')
                    fn_ratio_file = os.path.join(self.model_dir, 'fn_ratio.dat')
                    if epoch_id == 0:
                        with open(num_path_types_file, 'w') as o_f:
                            o_f.write('{}\n'.format(self.num_path_types))
                        with open(dev_metrics_file, 'w') as o_f:
                            o_f.write('{}\n'.format(metrics))
                        with open(hit_ratio_file, 'w') as o_f:
                            o_f.write('{}\n'.format(hit_ratio))
                        with open(fn_ratio_file, 'w') as o_f:
                            o_f.write('{}\n'.format(fn_ratio))
                    else:
                        with open(num_path_types_file, 'a') as o_f:
                            o_f.write('{}\n'.format(self.num_path_types))
                        with open(dev_metrics_file, 'a') as o_f:
                            o_f.write('{}\n'.format(metrics))
                        with open(hit_ratio_file, 'a') as o_f:
                            o_f.write('{}\n'.format(hit_ratio))
                        with open(fn_ratio_file, 'a') as o_f:
                            o_f.write('{}\n'.format(fn_ratio))

    def merge_abs_real_path_score(self, real_paths, entity_abs_paths):
        '''
        规则，如果一条real_paths的路径能跟entity_abs_paths对应的type的路径对应上，就merge
        :param real_paths: 真实的entity的paths
        :param entity_abs_paths: Abs和real一起走的paths
        :return:
        '''

        pass

    def forward(self, examples, abs_graph=False, same_infer=False, verbose=False):
        pred_scores = []
        for example_id in tqdm(range(0, len(examples), self.batch_size)):
            mini_batch = examples[example_id:example_id + self.batch_size]
            mini_batch_size = len(mini_batch)
            if len(mini_batch) < self.batch_size:

                self.make_full_batch(mini_batch, self.batch_size)
            if self.use_abstract_graph and same_infer:
                pred_score = self.predict_same(mini_batch, verbose=verbose)
            elif self.use_abstract_graph and abs_graph:
                # print("==>forward batch_size infer:", self.batch_size)
                pred_score = self.predict_abs(mini_batch, verbose=verbose)
            else:
                pred_score = self.predict(mini_batch, verbose=verbose)
            pred_scores.append(pred_score[:mini_batch_size])
        scores = torch.cat(pred_scores)
        return scores

    def format_batch(self, batch_data, num_labels=-1, num_tiles=1):
        """
        Convert batched tuples to the tensors accepted by the NN.
        """

        def convert_to_binary_multi_subject(e1):
            e1_label = zeros_var_cuda([len(e1), num_labels])
            for i in range(len(e1)):
                e1_label[i][e1[i]] = 1
            return e1_label

        def convert_to_binary_multi_object(e2):
            e2_label = zeros_var_cuda([len(e2), num_labels])
            for i in range(len(e2)):
                e2_label[i][e2[i]] = 1
            return e2_label

        batch_e1, batch_e2, batch_r = [], [], []
        for i in range(len(batch_data)):
            e1, e2, r = batch_data[i]
            batch_e1.append(e1)
            batch_e2.append(e2)
            batch_r.append(r)
        batch_e1 = var_cuda(torch.LongTensor(batch_e1), requires_grad=False)
        batch_r = var_cuda(torch.LongTensor(batch_r), requires_grad=False)
        if type(batch_e2[0]) is list:
            batch_e2 = convert_to_binary_multi_object(batch_e2)
        elif type(batch_e1[0]) is list:
            batch_e1 = convert_to_binary_multi_subject(batch_e1)
        else:
            batch_e2 = var_cuda(torch.LongTensor(batch_e2), requires_grad=False)
        # Rollout multiple times for each example
        if num_tiles > 1:
            batch_e1 = ops.tile_along_beam(batch_e1, num_tiles)
            batch_r = ops.tile_along_beam(batch_r, num_tiles)
            batch_e2 = ops.tile_along_beam(batch_e2, num_tiles)
        return batch_e1, batch_e2, batch_r

    def format_batch_with_abs(self, batch_data, num_labels=-1, num_tiles=1):
        """
        Convert batched tuples to the tensors accepted by the NN.
        """

        def convert_to_binary_multi_subject(e1):
            e1_label = zeros_var_cuda([len(e1), num_labels])
            e1_label_abs = zeros_var_cuda([len(e1), num_labels])
            for i in range(len(e1)):
                e1_label[i][e1[i]] = 1
                e1_label_abs[i][self.kg.get_typeid(e1[i])] = 1
            return e1_label, e1_label_abs

        def convert_to_binary_multi_object(e2):
            e2_label = zeros_var_cuda([len(e2), num_labels])
            e2_label_abs = zeros_var_cuda([len(e2), num_labels])
            for i in range(len(e2)):
                e2_label[i][e2[i]] = 1
                e2_label_abs[i][self.kg.get_typeid(e2[i])] = 1
            return e2_label, e2_label_abs

        batch_e1, batch_e2, batch_r, batch_e1_abs, batch_e2_abs, batch_r_abs = [], [], [], [], [], []
        for i in range(len(batch_data)):
            e1, e2, r = batch_data[i]
            batch_e1.append(e1)
            batch_e2.append(e2)
            batch_r.append(r)

            batch_e1_abs.append(self.kg.get_typeid(e1))
            batch_e2_abs.append(self.kg.get_typeid(e2))
            batch_r_abs.append(r)

        batch_e1 = var_cuda(torch.LongTensor(batch_e1), requires_grad=False)
        batch_r = var_cuda(torch.LongTensor(batch_r), requires_grad=False)
        batch_e1_abs = var_cuda(torch.LongTensor(batch_e1_abs), requires_grad=False)
        batch_r_abs = var_cuda(torch.LongTensor(batch_r_abs), requires_grad=False)
        if type(batch_e2[0]) is list:
            batch_e2, batch_e2_abs = convert_to_binary_multi_object(batch_e2)
        elif type(batch_e1[0]) is list:
            batch_e1, batch_e1_abs = convert_to_binary_multi_subject(batch_e1)
        else:
            batch_e2 = var_cuda(torch.LongTensor(batch_e2), requires_grad=False)
            batch_e2_abs = var_cuda(torch.LongTensor(batch_e2_abs), requires_grad=False)
        # Rollout multiple times for each example
        if num_tiles > 1:
            batch_e1 = ops.tile_along_beam(batch_e1, num_tiles)
            batch_r = ops.tile_along_beam(batch_r, num_tiles)
            batch_e2 = ops.tile_along_beam(batch_e2, num_tiles)
            batch_e1_abs = ops.tile_along_beam(batch_e1_abs, num_tiles)
            batch_r_abs = ops.tile_along_beam(batch_r_abs, num_tiles)
            batch_e2_abs = ops.tile_along_beam(batch_e2_abs, num_tiles)
        return batch_e1, batch_e2, batch_r, batch_e1_abs, batch_e2_abs, batch_r_abs

    def make_full_batch(self, mini_batch, batch_size, multi_answers=False):
        dummy_e = self.kg.dummy_e
        dummy_r = self.kg.dummy_r
        if multi_answers:
            dummy_example = (dummy_e, [dummy_e], dummy_r)
        else:
            dummy_example = (dummy_e, dummy_e, dummy_r)
        for _ in range(batch_size - len(mini_batch)):
            mini_batch.append(dummy_example)

    def save_checkpoint(self, checkpoint_id, epoch_id=None, is_best=False):
        """
        Save model checkpoint.
        :param checkpoint_id: Model checkpoint index assigned by training loop.
        :param epoch_id: Model epoch index assigned by training loop.
        :param is_best: if set, the model being saved is the best model on dev set.
        """
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = self.state_dict()
        checkpoint_dict['epoch_id'] = epoch_id

        out_tar = os.path.join(self.model_dir, 'checkpoint-{}.tar'.format(checkpoint_id))
        if is_best:
            best_path = os.path.join(self.model_dir, 'model_best.tar')
            shutil.copyfile(out_tar, best_path)
            print('=> best model updated \'{}\''.format(best_path))
        else:
            torch.save(checkpoint_dict, out_tar)
            print('=> saving checkpoint to \'{}\''.format(out_tar))

    def load_checkpoint(self, input_file):
        """
        Load model checkpoint.
        :param n: Neural network module.
        :param kg: Knowledge graph module.
        :param input_file: Checkpoint file path.
        """
        if os.path.isfile(input_file):
            print('=> loading checkpoint \'{}\''.format(input_file))
            print ("MAP GPU FROM 0 to 1")
            checkpoint = torch.load(input_file, map_location={
                'cuda:0': 'cuda:1'
            })
            self.load_state_dict(checkpoint['state_dict'])
            if not self.inference:
                self.start_epoch = checkpoint['epoch_id'] + 1
                assert (self.start_epoch <= self.num_epochs)
        else:
            print('=> no checkpoint found at \'{}\''.format(input_file))

    def export_to_embedding_projector(self):
        """
        Export knowledge base embeddings into .tsv files accepted by the Tensorflow Embedding Projector.
        """
        vector_path = os.path.join(self.model_dir, 'vector.tsv')
        meta_data_path = os.path.join(self.model_dir, 'metadata.tsv')
        v_o_f = open(vector_path, 'w')
        m_o_f = open(meta_data_path, 'w')
        for r in self.kg.relation2id:
            if r.endswith('_inv'):
                continue
            r_id = self.kg.relation2id[r]
            R = self.kg.relation_embeddings.weight[r_id]
            r_print = ''
            for i in range(len(R)):
                r_print += '{}\t'.format(float(R[i]))
            v_o_f.write('{}\n'.format(r_print.strip()))
            m_o_f.write('{}\n'.format(r))
            print(r, '{}'.format(float(R.norm())))
        v_o_f.close()
        m_o_f.close()
        print('KG embeddings exported to {}'.format(vector_path))
        print('KG meta data exported to {}'.format(meta_data_path))

    @property
    def rl_variation_tag(self):
        parts = self.model.split('.')
        if len(parts) > 1:
            return parts[1]
        else:
            return ''
