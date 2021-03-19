#!/bin/bash

# tfrecord数据在obs上存放的目录
BASE_CDN_DATA_DIR="obs://sprs-data-sg/sharezone_cdn/root/ads_train_data"

# 线上模型在obs上存放的目录
#BASE_ONLINE_DICT_DIR="obs://sprs-data-sg/sharezone_model_server/online_dict/deep_models"
BASE_ONLINE_DICT_DIR="s3://sprs.push.us-east-1.prod/data/warehouse/model/online_dict/deep_models"

# 模型上线的名称 - 推送模型上线时 或者 写监控数据时 必须配置
ONLINE_DICT_NAME="dnn_model_t1"

# 线上模型在本地存放的目录
BASE_ONLINE_DICT_LOCAL_DIR="/root/tf_trainer/online_dict/ads_effective"

# 在线预测时模型不需要的特征列表, 多个特征用逗号分隔
ONLINE_BLACK_FEATURES=""

# 是否开启共享embeddings(True or False)
IS_SHARE_EMB="False"

# 模型所用特征列表, 必须放到./data/目录
FEATURE_MAP_NAME="ads_feature_v3.csv"
#FEATURE_MAP_NAME="scene3_content.csv"

# run.sh所在的相对目录 - 以tf_trainer为根目录
RUN_DIR="./work/weshare/"

# new features embedding conf [field_name size]
#NEW_FEATURE_PATH="/tf/tensorflow/tf_trainer/work/weshare/data/game_h5_new.txt"
#OUT_EMBEDDING_PATH="/tf/tensorflow/tf_trainer/work_dir/ads_effective/increment_model/model.ckpt-23590"
#WILL_TRANSFER_EMBEDDING_PATH="/tf/tensorflow/tf_trainer/work_dir/ads_effective/20200810/summaries/default_game_h5_model_sample_v3/model.ckpt-23590"

# 执行run.sh相关任务的工作目录, 用于存放模型、日志等
WORK_DIR="/root/tf_trainer/work_dir/ads_effective/"

# 是否使用增量训练模式
USE_INCREMENT_MODE="True"

# 增量模型所在的本地目录, 若模型非增量设置为空即可
INCREMENT_MODEL_DIR="/root/tf_trainer/work_dir/ads_effective/increment_model"

# 生成model_server所用特征词典的脚本
# 非增量模型使用gen_feature_map_total.py, 增量模型使用gen_feature_map.py
FEATURE_INDEX_SCRIPT="gen_feature_map.py"

# 预测输出的列格式, 一般情况下用以下默认的就行
# 1. 预测输出时带上用户类型: label,user_id,user_type,pctr
PRED_OUTPUT_FORMAT="label,pctr"

# 用于分维度评估的样本标签, 配置格式为: [写入tfrecord中的key]:[key对应的特征类型], 多个标签用逗号分隔
# 默认配置中的group_id对应user_id
#DEBUG_LABELS="group_id:int,item_id:int,duration:int,user_level:int"
DEBUG_LABELS="is_click"

MULTI_TASK_CONFIG="is_effective_4:classification,play_duration:regression"

# tfrecord中用作label的key名称
LABEL_KEY="is_click"

# tfrecord中label的数据类型, 默认为float, 可选为int
LABEL_DTYPE='int'

# 是否使用无hash词典模式产出模型
DNN_WITH_HASH="False"

# 模型训练/测试所用的gpu设备号, 注意执行任务前先用nvidia-smi查看资源状态
CUDA_DEVICE_ID=1

# 模型种类, 参考tf_trainer/models/__init__.py中的条件判断
MODEL="default"

# 网络种类, 参考tf_trainer/networks/__init__.py中的network_factory函数
NETWORK="deepfm"

# dataset的种类, 参考tf_trainer/dataset/__init__.py中的build_dataset函数
DATASET="rec"

# attation配置, 格式为"特征名:特征名", 多个组合用逗号分割
ATTENTION_COLS="real_jclick_list_seq:item_id"

# 模型训练的任务名称
# 若多个训练任务共享同个WORKD_DIR, 可以设置不同的TASK_NAME, 方便查看对应的summary
TASK_NAME="ads_effective"

### training parameters ###
# 训练样本的迭代轮数
EPOCH_NUM=1

# 各个deep网络层的大小
DEEP_LAYERS="1024,512,256"

# 学习率
LEARNING_RATE=0.001

# tfrecord数据中一条记录的样本数
PREBATCH=1

# 训练时的batch
BATCH_SIZE=1024

# embedding向量的维度
EMB_DIM=8

### ssd - 表示数据下载到ssd上
### mhd - 表示数据下载到mhd上
WORK_ON_SSD_MHD="mhd"

MHD_PATH=""        # 机械盘根目录
SSD_PATH="/data2"       # 固态盘根目录

### online model verification threshold
VER_THRESH_AUC=-0.50                     # 模型离线auc最低值
VER_THRESH_GAUC=-1.0                    # 模型离线guac最低值
VER_THRESH_CALIBRATION_UPPER=200.0        # 模型离线calibration上限
VER_THRESH_CALIBRATION_LOWER=-0.8        # 模型离线calibration下限

# 模型最低size, 单位为MB
VER_THRESH_MODEL_SIZE=10

## Mail Notice
MAIL_ADDRESS="luojl@ushareit.com"
MAIL_SUBJECT="ads_effective Model Updating"
