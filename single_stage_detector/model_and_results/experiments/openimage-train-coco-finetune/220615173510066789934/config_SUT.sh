#!/usr/bin/env bash

## DL params
export BATCHSIZE=16
export NUMEPOCHS=30
export DATASET_DIR="/datasets/open-images-v6-mlperf"

# Trained with Coco, Finetuning with OpenImages 
# export EXTRA_PARAMS='--lr 0.0001 --output-dir=/results --dataset "openimages-mlperf" --resume-from-diff-dataset  --resume "/model/coco_epoch19_retinanet_resnext50_32x4d_fpn.pth"'

# Trained with Coco, Finetuning with OpenImages, Resume Finetuning
# export EXTRA_PARAMS='--lr 0.0001 --output-dir=/results --dataset=openimages-mlperf --resume=/model/coco_train_openimage_finetune_model_1.pth'

# Testing, just evulation
# export EXTRA_PARAMS='--eval-batch-size 4 --output-dir=/results --test-only --data-path=/datasets/open-images-v6-mlperf --resume=/model/coco_train_openimage_finetune_model_1.pth'

# Trained with OpenImages, Finetuning with Coco
# export EXTRA_PARAMS='--target-map=0.37 --lr 0.00005 --output-dir=/results --dataset=coco --resume=/model/openimage_model_10.pth --resume-from-diff-dataset --warmup-epochs=2'
export EXTRA_PARAMS='--target-map=0.37 --lr 0.0001 --output-dir=/results --dataset=coco --resume=/model/openimage_model_10.pth --resume-from-diff-dataset --warmup-epochs=2'

# Trained with OpenImages, Finetuning with Coco, Resume Finetuning
# export EXTRA_PARAMS='--target-map=0.37 --lr=0.0001 --output-dir=/results --dataset=coco --resume=/model/openimage_train_coco_finetune_model_12.pth'

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=08:00:00

## System config params
export DGXNGPU=2
export DGXSOCKETCORES=24
export DGXNSOCKET=1
export DGXHT=2  # HT is on is 2, HT off is 1
