#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training on midicn Dataset...'
    python train.py \
        --dataset midicn \
        --data_corpus txt \
        --max_step 120000 \
        --max_eval_steps 160 \
        --lr 0.0001 \
        --log_interval 50 \
        --eval_interval 1000 \
        --use_cuda \
        --multi_gpu \
        --gpu0_bsz 2 \
        --work_dir {DIR_NAME} \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --model_dir ./exp/midicn/{DIR_NAME}/ \
        --dataset midicn \
        --data_corpus txt \
        --split all \
        --use_cuda \
        ${@:2}
elif [[ $1 == 'gen' ]]; then
    echo 'Run generation...'
    python generate.py \
        --model_dir ./exp/midicn/{DIR_NAME}/ \
        --gen_dir gen \
        ${@:2}
else
    echo 'unknown argment 1'
fi