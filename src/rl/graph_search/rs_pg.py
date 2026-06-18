"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Policy gradient with reward shaping.
"""

from tqdm import tqdm

import torch

from src.emb.fact_network import get_conve_nn_state_dict, get_conve_kg_state_dict, \
    get_complex_kg_state_dict, get_distmult_kg_state_dict
from src.rl.graph_search.pg import PolicyGradient
import src.utils.ops as ops
from src.utils.ops import zeros_var_cuda

import src.rl.graph_search.beam_search as search
from src.utils.ops import int_fill_var_cuda, var_cuda, var_to_numpy, zeros_var_cuda

from src.rules.rule import HornRule
import numpy as np



import os
import matplotlib.pyplot as plt

class RewardShapingPolicyGradient(PolicyGradient):
    def __init__(self, args, kg, pn, fn_kg, fn, fn_secondary_kg=None):
        super(RewardShapingPolicyGradient, self).__init__(args, kg, pn)
        self.reward_shaping_threshold = args.reward_shaping_threshold

        # Fact network modules
        self.fn_kg = fn_kg
        self.fn = fn
        self.fn_secondary_kg = fn_secondary_kg
        self.mu = args.mu
        self.use_state_prediction = args.use_state_prediction
        self.hits = 0.0
        self.num = 0.0
        self.strategy = args.strategy

        fn_model = self.fn_model
        if fn_model in ['conve']:
            fn_state_dict = torch.load(args.conve_state_dict_path)
            fn_nn_state_dict = get_conve_nn_state_dict(fn_state_dict) #load params
            fn_kg_state_dict = get_conve_kg_state_dict(fn_state_dict)
            self.fn.load_state_dict(fn_nn_state_dict)
        elif fn_model == 'distmult':
            fn_state_dict = torch.load(args.distmult_state_dict_path)
            fn_kg_state_dict = get_distmult_kg_state_dict(fn_state_dict)
        elif fn_model == 'complex':
            fn_state_dict = torch.load(args.complex_state_dict_path)
            fn_kg_state_dict = get_complex_kg_state_dict(fn_state_dict)
        elif fn_model == 'hypere':
            fn_state_dict = torch.load(args.conve_state_dict_path)
            fn_kg_state_dict = get_conve_kg_state_dict(fn_state_dict)
        else:
            raise NotImplementedError
        self.fn_kg.load_state_dict(fn_kg_state_dict)
        if fn_model == 'hypere':
            complex_state_dict = torch.load(args.complex_state_dict_path)
            complex_kg_state_dict = get_complex_kg_state_dict(complex_state_dict)
            self.fn_secondary_kg.load_state_dict(complex_kg_state_dict)

        self.fn.eval()
        self.fn_kg.eval()
        ops.detach_module(self.fn)
        ops.detach_module(self.fn_kg)
        if fn_model == 'hypere':
            self.fn_secondary_kg.eval()
            ops.detach_module(self.fn_secondary_kg)

    #Rs
    def reward_fun(self, e1, r, e2, pred_e2):
        if self.model.endswith('.rso'):
            oracle_reward = forward_fact_oracle(e1, r, pred_e2, self.kg)
            return oracle_reward
        else:
            if self.fn_secondary_kg:
                real_reward = self.fn.forward_fact(e1, r, pred_e2, self.fn_kg, [self.fn_secondary_kg]).squeeze(1)
            else:
                real_reward = self.fn.forward_fact(e1, r, pred_e2, self.fn_kg).squeeze(1)
            real_reward_mask = (real_reward > self.reward_shaping_threshold).float()
            real_reward *= real_reward_mask
            if self.model.endswith('rsc'):
                return real_reward
            else:
                binary_reward = (pred_e2 == e2).float()
                # binary_reward = torch.gather(e2, 1, pred_e2.unsqueeze(-1)).squeeze(-1).float()

                return binary_reward + self.mu * (1 - binary_reward) * real_reward

    def conf_fun(self,path_trace,e1,r,e2):
        paths = [[] for i in range(len(path_trace[0][0]))]
        confs = []
        for step in range(len(path_trace)):
            for idx in range(len(path_trace[0][0])):
                paths[idx].append((path_trace[step][0][idx],path_trace[step][1][idx]))
        lazy_load_cnt = 0
        for idx in range(len(paths)):
            path = paths[idx]
            rule = HornRule(path,(int(e1[idx]),int(r[idx]),path[-1][1]),mode="CONSTANT_HEAD")
            if rule.get_str_representation() in self.rule2conf:
                conf = self.rule2conf[rule.get_str_representation()]
                lazy_load_cnt += 1
            else:
                path_trees = self.kg.cnt_paths_by_rule(rule,[int(e1[idx])])
                # path_trees = self.cnt_paths_by_rule(rule,[int(e1[idx])])
                conf_e1 = []
                conf_e2 = []
                conf_r = []
                conf_pe2 = []
                times = []
                for (pe2,weight) in path_trees.items():
                    # for i in range(len(leaves)):
                    #     if leaves[i] <= 0:
                    #         continue
                    conf_e1.append(int(e1[idx]))
                    conf_e2.append(int(e2[idx]))
                    conf_r.append(int(r[idx]))
                    conf_pe2.append(pe2)
                    if self.support_times:
                        times.append(weight)
                    else:
                        times.append(1)
                rewards = self.reward_fun(
                    var_cuda(torch.LongTensor(conf_e1), requires_grad=False),
                    var_cuda(torch.LongTensor(conf_r), requires_grad=False),
                    var_cuda(torch.LongTensor(conf_e2), requires_grad=False),
                    var_cuda(torch.LongTensor(conf_pe2), requires_grad=False)
                )
                rewards = var_to_numpy(rewards)
                conf = np.sum(rewards*times)/(np.sum(times)+1e-6)
                self.rule2conf[rule.get_str_representation()] = conf

            confs.append(conf)
        return var_cuda(torch.FloatTensor(confs), requires_grad=False)

    def cnt_paths_by_rule(self,rule,start_points,where="train"):
        # path_trees = {}
        if where == "all":
            space = self.kg.all_objects
        elif where == "dev":
            space = self.kg.dev_objects
        else:
            space = self.kg.adj_list
        for e1 in start_points:
            if self.args.path_search_policy == "tree":
                # leaves = self.search_adj_by_rule(e1,rule,0,space)
                re = self.search_adj_by_rule_no_recursion(e1,rule,space)
            elif self.args.path_search_policy == "matrix":
                leaves = self.search_adj_by_rule_cuda(e1,rule) #not complete
            else:
                raise RuntimeError("illegal path search policy")
            # path_trees[e1] = leaves
        return re

    def search_adj_by_rule_no_recursion(self,e1,rule,space):
        current_e = {}
        if e1 not in space:
            return {}
        current_e[e1] = 1
        for r in rule.get_rel_path():
            batch_r = var_cuda(torch.LongTensor([r]), requires_grad=False)
            next_e = {}
            for e,weight in current_e.items():
                if r in space[e]:
                    k = len(space[e][r])
                    batch_e = var_cuda(torch.LongTensor([e]), requires_grad=False)
                    S = self.fn.forward(batch_e, batch_r, self.fn_kg)  # len(e)*K_num
                    Sx, idx = torch.topk(S, k, dim=1)
                    for e2 in idx[0]:
                        e2_int=e2.item()
                        if e2_int not in next_e:
                            next_e[e2_int] = 0
                        next_e[e2_int] += weight
            current_e = next_e
        return current_e

    # def search_adj_by_rule_no_recursion(self,e1,rule,space):
    #     current_e = {}
    #     if e1 not in space:
    #         return {}
    #     # batch_e1 = var_cuda(torch.LongTensor([e1]), requires_grad=False)
    #     # batch_e1.squeeze_()
    #     current_e[e1] = 1
    #     for r in rule.get_rel_path():
    #         batch_r = var_cuda(torch.LongTensor([r]), requires_grad=False)
    #         next_e = {}
    #         for e,weight in current_e.items():
    #             batch_e = var_cuda(torch.LongTensor([e]), requires_grad=False)
    #             S = self.fn.forward(batch_e, batch_r, self.fn_kg)  # len(e)*K_num
    #             Sx, idx = torch.topk(S, k=3, dim=1)
    #             for e2 in idx[0]:
    #                 e2_int=e2.item()
    #                 if e2_int not in next_e:
    #                     next_e[e2_int] = 0
    #                 next_e[e2_int] += weight
    #         current_e = next_e
    #     return current_e

    def loss(self, mini_batch):

        def stablize_reward(r):
            r_2D = r.view(-1, self.num_rollouts)
            if self.baseline == 'avg_reward':
                stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
            elif self.baseline == 'max_min_scalar':
                stabled_r_2D = (r_2D - torch.min(r_2D,dim=1, keepdim=True)[0])/(torch.max(r_2D,dim=1, keepdim=True)[0] - torch.min(r_2D,dim=1, keepdim=True)[0])
            elif self.baseline == 'avg_reward_normalized':
                stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (r_2D.std(dim=1, keepdim=True) + ops.EPSILON)
            else:
                raise ValueError('Unrecognized baseline function: {}'.format(self.baseline))
            stabled_r = stabled_r_2D.view(-1)
            return stabled_r

        #list of (e1,e2,r),size of batchsize
        # e1, e2, r, kg_pred = self.format_batch(mini_batch, num_labels=self.kg.num_entities, num_tiles=self.num_rollouts)
        e1, e2, r, kg_pred = self.format_batch(mini_batch, num_tiles=self.num_rollouts)
        #here become 1 dim temsor,size of batchsize*num_rollouts
        # output = self.rollout(e1, r, e2, num_steps=self.num_rollout_steps)
        output = self.rollout(e1, r, e2, num_steps=self.num_rollout_steps, kg_pred=kg_pred)


        # Compute policy gradient loss
        pred_e2 = output['pred_e2']
        log_action_probs = output['log_action_probs']
        action_entropy = output['action_entropy']
        path_trace = output["path_trace"]
        #path_trace(steps, (r,e), batch_size)

        baseline_reward = self.reward_fun(e1, r, e2, pred_e2)
        if self.use_conf:
            final_reward = self.conf_fun(
                [(var_to_numpy(r),var_to_numpy(e)) for (r,e) in path_trace],
                var_to_numpy(e1),
                var_to_numpy(r),
                var_to_numpy(e2)
            )
        else:
            final_reward = self.reward_fun(e1, r, e2, pred_e2)

        if self.baseline == 'n/a':
            final_reward = torch.where(final_reward<baseline_reward,final_reward,baseline_reward)
        elif self.baseline == "curriculum":
            final_reward = self.alpha * final_reward + (1-self.alpha) * baseline_reward
        else:
            final_reward = stablize_reward(final_reward)

        # if self.plot:
        #     x = var_to_numpy(baseline_reward)
        #     y = var_to_numpy(final_reward)
        #     x = x - 0.5
        #     y = y - 0.5
        #     plt.cla()
        #     plt.scatter(x,y)
        #     ax = plt.gca()
        #     ax.spines['right'].set_color('r')
        #     ax.spines['top'].set_color('none')
        #     ax.xaxis.set_ticks_position('bottom')
        #     ax.spines['bottom'].set_position(('data',0))
        #     ax.yaxis.set_ticks_position('left')
        #     ax.spines['left'].set_position(('data',0))
        #     plt.savefig(os.path.join(self.model_dir,"dist{}.png".format(self.plotid)))
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

    def format_batch(self, batch_data, num_labels=-1, num_tiles=1, inference=False):
        """
        Convert batched tuples to the tensors accepted by the NN.
        """
        def convert_to_binary_multi_subject(e1):
            e1_label = zeros_var_cuda([len(e1), num_labels])
            for i in range(len(e1)):
                e1_label[i][e1[i]] = 1
            return e1_label

        def convert_to_binary_multi_object(e2):
            # e2_label = zeros_var_cuda([len(e2), self.kg.num_entities])
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
        if inference:
            batch_e2_new = []
            for i in range(len(batch_e1)):
                tmp = []
                if batch_e1[i] in self.kg.train_objects and batch_r[i] in self.kg.train_objects[batch_e1[i]]:
                    tmp += list(self.kg.train_objects[batch_e1[i]][batch_r[i]])
                batch_e2_new.append(tmp)
            batch_e2 = batch_e2_new
        batch_e1 = var_cuda(torch.LongTensor(batch_e1), requires_grad=False)
        batch_r = var_cuda(torch.LongTensor(batch_r), requires_grad=False)
        if type(batch_e2[0]) is list:
            batch_e2 = convert_to_binary_multi_object(batch_e2)
        elif type(batch_e1[0]) is list:
            batch_e1 = convert_to_binary_multi_subject(batch_e1)
        else:
            batch_e2 = var_cuda(torch.LongTensor(batch_e2), requires_grad=False)
        # Rollout multiple times for each example

        # weight emb
        # S = self.fn.forward(batch_e1, batch_r, self.fn_kg).view(-1, self.kg.num_entities)
        # pred_kg = torch.matmul(S, self.kg.entity_embeddings.weight) / torch.sum(S, dim=1, keepdim=True)

        # sample emb
        if self.strategy == 'sample':
            # sample emb
            S = self.fn.forward(batch_e1, batch_r, self.fn_kg).view(-1, self.kg.num_entities)
            a = torch.multinomial(S, 1)
            pred_kg = self.kg.entity_embeddings(a.view(self.batch_size))
        elif self.strategy == 'avg':
            # weight emb
            S = self.fn.forward(batch_e1, batch_r, self.fn_kg).view(-1, self.kg.num_entities)
            pred_kg = torch.matmul(S, self.kg.entity_embeddings.weight) / torch.sum(S, dim=1, keepdim=True)
        elif self.strategy == 'top1':
            # Top K method
            S = self.fn.forward(batch_e1, batch_r, self.fn_kg)
            Sx, idx = torch.topk(S, k=1, dim=1)
            Sx = Sx.unsqueeze(-1)
            S = self.kg.entity_embeddings(idx) * Sx
            x = torch.sum(Sx, dim=1, keepdim=True)
            S = S / x
            pred_kg = torch.sum(S, dim=1)

        # hits = float(torch.sum(torch.gather(batch_e2, 1, a), dim=0))
        # self.hits += hits
        # self.num += float(a.shape[0])
        # print('Hits ratio: {}'.format(self.hits / self.num))

        # Top K method
        # S = self.fn.forward(batch_e1, batch_r, self.fn_kg)
        # Sx, idx = torch.topk(S, k=1, dim=1)
        # Sx = Sx.unsqueeze(-1)
        # S = self.kg.entity_embeddings(idx) * Sx
        # x = torch.sum(Sx, dim=1, keepdim=True)
        # S = S / x
        # pred_kg = torch.sum(S, dim=1)

        # Top K with fc
        # S = self.fn.forward(batch_e1, batch_r, self.fn_kg)
        # _, idx = torch.topk(S, k=10, dim=1)
        # pred_kg = self.fc1(self.kg.entity_embeddings(idx)).view(self.batch_size, -1)

        if num_tiles > 1:
            batch_e1 = ops.tile_along_beam(batch_e1, num_tiles)
            batch_r = ops.tile_along_beam(batch_r, num_tiles)
            batch_e2 = ops.tile_along_beam(batch_e2, num_tiles)
            pred_kg = ops.tile_along_beam(pred_kg, num_tiles)
        return batch_e1, batch_e2, batch_r, pred_kg

    def rollout(self, e_s, q, e_t, num_steps, kg_pred, visualize_action_probs=False):
        """
        Perform multi-step rollout from the source entity conditioned on the query relation.
        :param pn: Policy network.
        :param e_s: (Variable:batch) source entity indices.
        :param q: (Variable:batch) query embedding.
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
            obs = [e_s, q, e_t, t==(num_steps-1), last_r, seen_nodes]
            db_outcomes, inv_offset, policy_entropy = pn.transit(
                e, obs, kg, kg_pred=kg_pred, fn_kg=self.fn_kg, use_action_space_bucketing=self.use_action_space_bucketing, use_kg_pred=self.use_state_prediction)
            sample_outcome = self.sample_action(db_outcomes, inv_offset)
            action = sample_outcome['action_sample']
            pn.update_path(action, kg)
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

    def predict(self, mini_batch, verbose=False):
        kg, pn = self.kg, self.mdl
        e1, e2, r, kg_pred = self.format_batch(mini_batch, inference=False)
        beam_search_output = search.beam_search(
            pn, e1, r, e2, kg, self.num_rollout_steps, self.beam_size, use_kg_pred=self.use_state_prediction, kg_pred=kg_pred, fn_kg=self.fn_kg)
        pred_e2s = beam_search_output['pred_e2s']
        pred_e2_scores = beam_search_output['pred_e2_scores']
        if verbose:
            # print inference paths
            # MAX_PATH = 10
            search_traces = beam_search_output['search_traces']
            output_beam_size = min(self.beam_size, pred_e2_scores.shape[1])
            # output_beam_size = min(self.beam_size, pred_e2_scores.shape[1], MAX_PATH)
            paths = []#
            for i in range(len(e1)):
                h = kg.id2entity[int(e1[i])]
                rel = kg.id2relation[int(r[i])]
                t = kg.id2entity[int(e2[i])]
                with open('mycase.txt', 'a') as file:
                    file.write('*****************************************************\n')
                    file.write('({}, {}, {})\n'.format(h, rel, t))
                # print('({}, {}, {})'.format(h, rel, t))
                beam = []#
                for j in range(output_beam_size):
                    ind = i * output_beam_size + j
                    if pred_e2s[i][j] == kg.dummy_e:
                        break
                    search_trace = []
                    for k in range(len(search_traces)):
                        search_trace.append((int(search_traces[k][0][ind]), int(search_traces[k][1][ind])))
                    beam.append(search_trace)#
                    with open('mycase.txt', 'a') as file:
                        file.write('<PATH> {} \n'.format(ops.format_path(search_trace, kg)))
                    # print('<PATH> {}'.format(ops.format_path(search_trace, kg)))
                with open('mycase.txt', 'a') as file:
                    file.write('*****************************************************\n')
                paths.append(beam)
        with torch.no_grad():
            pred_scores = zeros_var_cuda([len(e1), kg.num_entities])
            for i in range(len(e1)):
                pred_scores[i][pred_e2s[i]] = torch.exp(pred_e2_scores[i])
                re_paths = None
            if verbose:#
                re_paths = [[list() for j in range(kg.num_entities)] for i in range(len(e1))]
                for i in range(len(e1)):
                    for j in range(len(pred_e2s[i])):
                        if pred_e2s[i][j] == kg.dummy_e:
                            break
                        re_paths[i][pred_e2s[i][j]] = paths[i][j]
        return pred_scores,re_paths

    def test_fn(self, examples):
        fn_kg, fn = self.fn_kg, self.fn
        pred_scores = []
        for example_id in tqdm(range(0, len(examples), self.batch_size)):
            mini_batch = examples[example_id:example_id + self.batch_size]
            mini_batch_size = len(mini_batch)
            if len(mini_batch) < self.batch_size:
                self.make_full_batch(mini_batch, self.batch_size)
            e1, e2, r = self.format_batch(mini_batch)
            if self.fn_secondary_kg:
                pred_score = fn.forward_fact(e1, r, e2, fn_kg, [self.fn_secondary_kg])
            elif self.fn_model == 'PTransE':
                # TODO
                pred_score = fn.forward_fact(e1, r, e2, fn_kg)
            else:
                pred_score = fn.forward_fact(e1, r, e2, fn_kg)
            pred_scores.append(pred_score[:mini_batch_size])
        return torch.cat(pred_scores)

    @property
    def fn_model(self):
        return self.model.split('.')[3] if self.model.split('.')[2] == "new" else self.model.split('.')[2]

def forward_fact_oracle(e1, r, e2, kg):
    oracle = zeros_var_cuda([len(e1), kg.num_entities]).cuda()
    for i in range(len(e1)):
        _e1, _r = int(e1[i]), int(r[i])
        if _e1 in kg.all_object_vectors and _r in kg.all_object_vectors[_e1]:
            answer_vector = kg.all_object_vectors[_e1][_r]
            oracle[i][answer_vector] = 1
        else:
            raise ValueError('Query answer not found')
    oracle_e2 = ops.batch_lookup(oracle, e2.unsqueeze(1))
    return oracle_e2
