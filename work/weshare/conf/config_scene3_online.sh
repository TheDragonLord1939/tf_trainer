#!/bin/bash

BASE_CDN_DATA_DIR="obs://sprs-data-sg/sharezone_cdn/weianjun/online/"

BASE_ONLINE_DICT_LOCAL_DIR="/data2/weianjun/online_dict/scene3/"

BASE_ONLINE_DICT_DIR="obs://sprs-data-sg/sharezone_model_server/online_dict/deep_models/"

ONLINE_DICT_NAME="scene3_dnn_model"

FEATURE_MAP_NAME="scene3_online.csv"

RUN_DIR="./work/weshare"
WORK_DIR="/data2/weianjun/online_work_dir/scene3"

INCREMENT_MODEL_DIR="/data2/weianjun/online_work_dir/scene3/increment_model"

FEATURE_INDEX_SCRIPT="gen_feature_map.py"

#TRAIN_LABEL,DEBUG_LABELS,PRED
PRED_OUTPUT_FORMAT="label,user_id,pctr"

DEBUG_LABELS="group_id:int"

LABEL_KEY="is_click"

LABEL_DTYPE="int"

CUDA_DEVICE_ID=5

DNN_WITH_HASH="False"

IS_POSTERIOR_VALID="True"

PIPELINE_CLEAN_DATA="True"

#ATTENTION_COLS="real_click_list_seq:item_id"

MODEL="default"
NETWORK="deepfm"
DATASET="rec"
ATTENTION_COLS=""
TASK_NAME="online"

### training parameters ###
EPOCH_NUM=1
DEEP_LAYERS="400,400,400"
LEARNING_RATE=0.001
PREBATCH=1024
BATCH_SIZE=16
USE_INCREMENT_MODE="True"

### ssd - 表示数据下载到ssd上
### mhd - 表示数据下载到mhd上
WORK_ON_SSD_MHD="ssd"
MHD_PATH="/data"
SSD_PATH="/data2"

### online model verification threshold
VER_THRESH_AUC=0.67
VER_THRESH_GAUC=0.55
VER_THRESH_CALIBRATION_UPPER=1.2
VER_THRESH_CALIBRATION_LOWER=0.8

# 模型大小, 单位为B
VER_THRESH_MODEL_SIZE=739

## mail 
#MAIL_ADDRESS="weiaj@ushareit.com"
MAIL_ADDRESS="weiaj@ushareit.com,luojl@ushareit.com,jinch@ushareit.com,guorui@ushareit.com,tengfei@ushareit.com,xuhao@ushareit.com,sibb@ushareit.com,candy@ushareit.com,zhaoyq@ushareit.com"
MAIL_SUBJECT="PushRel Online DNN Incremental Model Updating"
