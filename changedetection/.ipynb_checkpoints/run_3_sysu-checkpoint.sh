#!/bin/bash

cd /root/MambaCD/changedetection

DATA_ROOT=/root/autodl-tmp/data/SYSU-CD
SAVE_ROOT=/root/autodl-tmp/saved_models

CFG=configs/vssm1/vssm_tiny_224_0229flex.yaml
PRETRAIN=/root/autodl-tmp/pretrained_weight/vssm_tiny_0229_ckpt_epoch_264.pth

mkdir -p ${SAVE_ROOT}
mkdir -p logs
RUNNING=$(ps -ef | grep train_MambaBCD.py | grep -v grep | wc -l)

if [ "$RUNNING" -gt 0 ]; then
  echo "ERROR: There are already $RUNNING training jobs running."
  echo "Please stop them first with:"
  echo "pkill -f train_MambaBCD.py"
  exit 1
fi
# =========================
# 1. Baseline: original ChangeMamba
# =========================
CUDA_VISIBLE_DEVICES=0 nohup python script/train_MambaBCD.py \
  --dataset SYSU \
  --type train \
  --train_dataset_path ${DATA_ROOT}/train \
  --train_data_list_path ${DATA_ROOT}/train.txt \
  --test_dataset_path ${DATA_ROOT}/test \
  --test_data_list_path ${DATA_ROOT}/test.txt \
  --model_type MambaBCD_Tiny_baseline \
  --model_param_path ${SAVE_ROOT}/SYSU_pixel_3407 \
  --cfg ${CFG} \
  --pretrained_weight_path ${PRETRAIN} \
  --batch_size 8 \
  --crop_size 256 \
  --max_iters 20000 \
  --learning_rate 1e-4 \
  --weight_decay 5e-3 \
  --momentum 0.9 \
  --eval_interval 500 \
  --gate_mode pixel \
  --dropout_rate 0.0 \
  --seed 3407 \
  > logs/SYSU_pixel_3407.log 2>&1 &


# =========================
# 2. Image-level dynamic gate
# =========================
CUDA_VISIBLE_DEVICES=1 nohup python script/train_MambaBCD.py \
  --dataset SYSU \
  --type train \
  --train_dataset_path ${DATA_ROOT}/train \
  --train_data_list_path ${DATA_ROOT}/train.txt \
  --test_dataset_path ${DATA_ROOT}/test \
  --test_data_list_path ${DATA_ROOT}/test.txt \
  --model_type MambaBCD_Tiny_image_gate \
  --model_param_path ${SAVE_ROOT}/SYSU_pixel_gate_uq01_seed3407 \
  --cfg ${CFG} \
  --pretrained_weight_path ${PRETRAIN} \
  --batch_size 8 \
  --crop_size 256 \
  --max_iters 20000 \
  --learning_rate 1e-4 \
  --weight_decay 5e-3 \
  --momentum 0.9 \
  --eval_interval 500 \
  --use_uncertainty \
  --uncertainty_weight 0.1 \
  --dropout_rate 0.0 \
  --seed 3407 \
  > logs/SYSU_pixel_gate_uq01_seed3407.log 2>&1 &


# =========================
# 3. Pixel-level gate + uncertainty
# =========================
CUDA_VISIBLE_DEVICES=2 nohup python script/train_MambaBCD.py \
  --dataset SYSU \
  --type train \
  --train_dataset_path ${DATA_ROOT}/train \
  --train_data_list_path ${DATA_ROOT}/train.txt \
  --test_dataset_path ${DATA_ROOT}/test \
  --test_data_list_path ${DATA_ROOT}/test.txt \
  --model_type MambaBCD_Tiny_pixel_gate_uq \
  --model_param_path ${SAVE_ROOT}/SYSU_pixel_gate_uq0001_seed3407 \
  --cfg ${CFG} \
  --pretrained_weight_path ${PRETRAIN} \
  --batch_size 8 \
  --crop_size 256 \
  --max_iters 20000 \
  --learning_rate 1e-4 \
  --weight_decay 5e-3 \
  --momentum 0.9 \
  --eval_interval 500 \
  --gate_mode pixel \
  --use_uncertainty \
  --uncertainty_weight 0.001 \
  --dropout_rate 0.0 \
  --seed 3407 \
  > logs/SYSU_pixel_gate_uq0001_seed3407.log 2>&1 &

