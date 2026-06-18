#!/usr/bin/env bash

data_dir="data/WD-singer"
model="point.rs.conve"
group_examples_by_query="False"
use_action_space_bucketing="True"

use_action_selection="True"
use_state_prediction="False"
mask_sim_relation="False"
max_dynamic_action_size=20
dynamic_split_bound=2
avg_entity_per_relation=1
strategy="sample"

use_conf="True"
support_times="True"
path_search_policy="tree"
baseline="curriculum"
baseline1="n/a"
decrease_step=5
decrease_rate=0.8
decrease_offline=0.1

bandwidth=400
entity_dim=200
relation_dim=200
history_dim=200
history_num_layers=3
num_rollouts=20
num_rollout_steps=3
bucket_interval=10
num_epochs=240
num_wait_epochs=10
num_peek_epochs=1
batch_size=96
train_batch_size=96
dev_batch_size=4
learning_rate=0.001
grad_norm=0
emb_dropout_rate=0.3
ff_dropout_rate=0.1
action_dropout_rate=0.5
action_dropout_anneal_interval=1000
reward_shaping_threshold=0
beta=0.02
relation_only="False"
save_beam_search_paths="True"
beam_size=128

distmult_state_dict_path="model/NELL23K-conve-RV-xavier-200-200-0.003-32-3-0.3-0.3-0.2-0.1/model_best.tar"
complex_state_dict_path="model/NELL23K-conve-RV-xavier-200-200-0.003-32-3-0.3-0.3-0.2-0.1/model_best.tar"
conve_state_dict_path="model/NELL23K-conve-RV-xavier-200-200-0.003-32-3-0.3-0.3-0.2-0.1/model_best.tar"
checkpoint_path1="model/NELL23K-conve-RV-xavier-200-200-0.003-32-3-0.3-0.3-0.2-0.1/model_best.tar"
checkpoint_path="None"

num_paths_per_entity=-1
margin=-1
