#!/bin/bash

BASE_CDN_DATA_DIR="obs://sprs-data-sg/sharezone_cdn/weianjun/push_base/"

BASE_ONLINE_DICT_LOCAL_DIR="/data2/weianjun/online_dict/push_base/"

BASE_ONLINE_DICT_DIR="obs://sprs-data-sg/push/online_dict/deep_models/"

ONLINE_DICT_NAME="push_dnn_model"

FEATURE_MAP_NAME="push.csv"

RUN_DIR="./work/weshare"

WORK_DIR="/data2/weianjun/online_work_dir/push_base"

INCREMENT_MODEL_DIR="/data2/weianjun/online_work_dir/push_base/increment_model"

#INCREMENT_MODEL_CLOUD_DIR="obs://sprs-data-sg/NeverDelete/weianjun/backup_model"

FEATURE_INDEX_SCRIPT="gen_feature_map.py"

#TRAIN_LABEL,DEBUG_LABELS,PRED
PRED_OUTPUT_FORMAT="label,user_id,bucket_id,recall_key,pctr"

DEBUG_LABELS="group_id:int,_bucket_id:int,_recall_key:int"

LABEL_KEY="is_click"

LABEL_DTYPE="int"

CUDA_DEVICE_ID=5

DNN_WITH_HASH="False"

#IS_SHARE_EMB="True"

#EMB_DIM=8

IS_POSTERIOR_VALID="True"

PIPELINE_CLEAN_DATA="True"

#ATTENTION_COLS="real_click_list_seq:item_id"

MODEL="default"
NETWORK="deepfm"
DATASET="rec"
ATTENTION_COLS=""
TASK_NAME="test"

### training parameters ###
EPOCH_NUM=1
DEEP_LAYERS="400,400,400"
LEARNING_RATE=0.001
PREBATCH=1024
BATCH_SIZE=10
USE_INCREMENT_MODE="True"

### ssd - 表示数据下载到ssd上
### mhd - 表示数据下载到mhd上
WORK_ON_SSD_MHD="ssd"
MHD_PATH="/data"
SSD_PATH="/data2"

### online model verification threshold
VER_THRESH_AUC=0.7
VER_THRESH_GAUC=0.5
VER_THRESH_CALIBRATION_UPPER=100
VER_THRESH_CALIBRATION_LOWER=0

# 模型大小, 单位为B
VER_THRESH_MODEL_SIZE=700

## mail 
#MAIL_ADDRESS="weiaj@ushareit.com"
MAIL_ADDRESS="weiaj@ushareit.com,tengfei@ushareit.com,sibb@ushareit.com,zhaosl@ushareit.com"
MAIL_SUBJECT="Push Base DNN Incremental Model Updating"
