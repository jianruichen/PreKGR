#!/bin/bash

export PYTHONPATH=`pwd`
echo $PYTHONPATH

source $1
exp=$2
gpu=$3
ARGS=${@:4}

group_examples_by_query_flag=''
if [[ $group_examples_by_query = *"True"* ]]; then
    group_examples_by_query_flag="--group_examples_by_query"
fi
relation_only_flag=''
if [[ $relation_only = *"True"* ]]; then
    relation_only_flag="--relation_only"
fi
use_action_space_bucketing_flag=''
if [[ $use_action_space_bucketing = *"True"* ]]; then
    use_action_space_bucketing_flag='--use_action_space_bucketing'
fi
use_conf_flag=''
if [[ $use_conf = *"True"* ]]; then
    use_conf_flag='--use_conf'
fi
support_times_flag=''
if [[ $support_times = *"True"* ]]; then
    support_times_flag='--support_times'
fi
use_action_selection_flag=''
if [[ $use_action_selection = *"True"* ]]; then
    use_action_selection_flag='--use_action_selection'
fi
use_state_prediction_flag=''
if [[ $use_state_prediction = *"True"* ]]; then
    use_state_prediction_flag='--use_state_prediction'
fi
mask_sim_relation_flag=''
if [[ $mask_sim_relation = *"True"* ]]; then
    mask_sim_relation_flag='--mask_sim_relation'
fi
save_beam_search_paths_flag=''
if [[ $save_beam_search_paths = *"True"* ]]; then
    save_beam_search_paths_flag='--save_beam_search_paths'
fi

cmd="python -m src.experiments \
    --data_dir $data_dir \
    $exp \
    --model $model \
    --bandwidth $bandwidth \
    --entity_dim $entity_dim \
    --relation_dim $relation_dim \
    --history_dim $history_dim \
    --history_num_layers $history_num_layers \
    --num_rollouts $num_rollouts \
    --num_rollout_steps $num_rollout_steps \
    --bucket_interval $bucket_interval \
    --num_epochs $num_epochs \
    --num_wait_epochs $num_wait_epochs \
    --num_peek_epochs $num_peek_epochs \
    --batch_size $batch_size \
    --train_batch_size $train_batch_size \
    --dev_batch_size $dev_batch_size \
    --margin $margin \
    --learning_rate $learning_rate \
    --baseline $baseline \
    --grad_norm $grad_norm \
    --emb_dropout_rate $emb_dropout_rate \
    --ff_dropout_rate $ff_dropout_rate \
    --action_dropout_rate $action_dropout_rate \
    --action_dropout_anneal_interval $action_dropout_anneal_interval \
    --reward_shaping_threshold $reward_shaping_threshold \
    --decrease_step $decrease_step \
    --decrease_rate $decrease_rate \
    --decrease_offline $decrease_offline \
    $relation_only_flag \
    --beta $beta \
    --beam_size $beam_size \
    --num_paths_per_entity $num_paths_per_entity \
    $group_examples_by_query_flag \
    $use_action_space_bucketing_flag \
    $use_conf_flag \
    $support_times_flag \
    --distmult_state_dict_path $distmult_state_dict_path \
    --complex_state_dict_path $complex_state_dict_path \
    --conve_state_dict_path $conve_state_dict_path \
    --checkpoint_path $checkpoint_path \
    --path_search_policy $path_search_policy \
    $use_action_selection_flag \
    $use_state_prediction_flag \
    $mask_sim_relation_flag \
    --max_dynamic_action_size $max_dynamic_action_size \
    --dynamic_split_bound $dynamic_split_bound \
    --avg_entity_per_relation $avg_entity_per_relation \
    --strategy $strategy \
    --gpu $gpu \
    $ARGS"

echo "Executing $cmd"

$cmd
