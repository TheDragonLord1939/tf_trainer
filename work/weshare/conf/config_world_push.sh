#!/bin/bash

#BASE_CDN_DATA_DIR="obs://sprs-data-sg/sharezone_cdn/abc/online/"
BASE_CDN_DATA_DIR="obs://sprs-data-sg/sharezone_model_server/ps_model/push_dnn/shareit/"

BASE_ONLINE_DICT_LOCAL_DIR="/data/abc/online_dict/id_push/"

BASE_ONLINE_DICT_DIR="obs://sprs-data-sg/push/online_dict/deep_models/"

ONLINE_DICT_NAME="world_push_dnn_model"

FEATURE_MAP_NAME="push_features.csv"

RUN_DIR="./work/weshare"

WORK_DIR="/data/abc/online_work_dir/push_base"

INCREMENT_MODEL_DIR="/data2/abc/online_work_dir/push_base/increment_model"

#INCREMENT_MODEL_CLOUD_DIR="obs://sprs-data-sg/NeverDelete/weianjun/backup_model"

FEATURE_INDEX_SCRIPT="gen_feature_map.py"

#TRAIN_LABEL,DEBUG_LABELS,PRED
PRED_OUTPUT_FORMAT="label,user_id,country,hour,pctr"

DEBUG_LABELS="user_id:int,country:int,hour:int"

LABEL_KEY="is_click"

LABEL_DTYPE="int"

CUDA_DEVICE_ID=7

DNN_WITH_HASH="False"

#IS_SHARE_EMB="True"

EMB_DIM=8

IS_POSTERIOR_VALID="False"

PIPELINE_CLEAN_DATA="True"

#ATTENTION_COLS="real_click_list_seq:item_id"

MODEL="default"
NETWORK="deepfm"
DATASET="rec"
ATTENTION_COLS="push_real_click_list:item_id,push_show_not_click_list:item_id"
TASK_NAME="test"

### training parameters ###
EPOCH_NUM=1
DEEP_LAYERS="512,256,256"
LEARNING_RATE=0.0005
PREBATCH=1024
BATCH_SIZE=10
USE_INCREMENT_MODE="True"

### ssd - 表示数据下载到ssd上
### mhd - 表示数据下载到mhd上
WORK_ON_SSD_MHD="ssd"
MHD_PATH="/data"
SSD_PATH="/data2"

### online model verification threshold
VER_THRESH_AUC=0.5
VER_THRESH_GAUC=0.5
VER_THRESH_CALIBRATION_UPPER=100
VER_THRESH_CALIBRATION_LOWER=0

# 模型大小, 单位为B
VER_THRESH_MODEL_SIZE=700

## mail
#MAIL_ADDRESS="weiaj@ushareit.com"
MAIL_ADDRESS="jiangda@ushareit.com,longxb@ushareit.com,mazd@ushareit.com,chenxk@ushareit.com,lianghb@ushareit.com"
MAIL_SUBJECT="Push Base DNN Incremental Model Updating"
