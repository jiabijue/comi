#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training on midicn Dataset...'
    CUDA_VISIBLE_DEVICES=0
    python main_rnn.py \
        --augment_pitch \
        --unit_type lstm \
        --use_cuda \
        --work_dir {DIR_NAME}
        ${@:2}
elif [[ $1 == 'gen' ]]; then
    echo 'Run generation...'
    python main_rnn.py \
        --mode gen \
        --dataset midicn \
        --unit_type lstm \
        --work_dir {DIR_NAME} \

    python main_rnn.py \
        --mode gen \
        --dataset midicn \
        --unit_type gru \
        --work_dir {DIR_NAME} \
        ${@:2}
else
    echo 'unknown argment 1'
fi