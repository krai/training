#/bin/bash

RANDOM_SEED=$1
QUALITY=$2
set -e

# Register the model as a source root
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Set variables
BATCH_SIZE=408 # Global Batch Size
NUM_GPUS=2
WARMUP_EPOCHS=2
TRAIN_EPOCHS=37
# STEPS_PER_LOOP=109935


# Precompute variables
ceildiv() {
    local num=$1
    local div=$2
    echo $(( (num + div - 1) / div ))
}

TRAIN_SAMPLES=1281167
MODEL_BN_SPAN=$(ceildiv ${BATCH_SIZE} ${NUM_GPUS})
NUM_SAMPLES_PER_GPU=$(ceildiv ${TRAIN_SAMPLES} ${NUM_GPUS})
STEPS_PER_LOOP=$(ceildiv ${NUM_SAMPLES_PER_GPU} ${MODEL_BN_SPAN})

python3 ./resnet_ctl_imagenet_main.py \
--base_learning_rate=3.0 \
--batch_size=${BATCH_SIZE} \
--data_dir="/data" \
--dtype=fp32 \
--device_warmup_steps=1 \
--noenable_device_warmup \
--enable_eager \
--noenable_xla \
--epochs_between_evals=1 \
--noeval_dataset_cache \
--eval_offset_epochs=0 \
--eval_prefetch_batchs=1 \
--label_smoothing=0.1 \
--lars_epsilon=0 \
--log_steps=125 \
--lr_schedule=polynomial \
--model_dir="/model"\
--momentum=0.9 \
--num_accumulation_steps=2 \
--num_classes=1000 \
--num_gpus=${NUM_GPUS} \
--optimizer=LARS \
--noreport_accuracy_metrics \
--single_l2_loss_op \
--noskip_eval \
--steps_per_loop=${STEPS_PER_LOOP} \
--target_accuracy=0.759 \
--notf_data_experimental_slack \
--tf_gpu_thread_mode=gpu_private \
--notrace_warmup \
--train_epochs=${TRAIN_EPOCHS} \
--notraining_dataset_cache \
--training_prefetch_batchs=1 \
--nouse_synthetic_data \
--warmup_epochs=${WARMUP_EPOCHS} \
--weight_decay=0.00005 \
--report_accuracy_metrics \
--enable_checkpoint_and_export
