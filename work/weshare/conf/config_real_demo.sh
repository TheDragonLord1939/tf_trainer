#!/bin/bash

#--------------- 数据存放和获取相关配置 ---------------#

# tfrecord数据在本地存放的根目录
BASE_DATA_DIR="/mnt/weianjun/workspace/work_dir/tfrecord"

# 线上模型在obs上存放的目录
# BASE_ONLINE_DICT_DIR="obs://sprs-data-sg/sharezone_model_server/online_dict/deep_models"
BASE_ONLINE_DICT_DIR="obs://sprs-data-sg/NeverDelete/weianjun/temp"

# 线上模型在本地存放的目录
BASE_ONLINE_DICT_LOCAL_DIR="/mnt/weianjun/online_dict"

# 执行run.sh相关任务的工作目录, 用于存放模型、日志等
WORK_DIR="/mnt/weianjun/workspace/work_dir"

# 增量模型所在的本地目录, 若模型非增量设置为空即可
INCREMENT_MODEL_DIR="/mnt/weianjun/workspace/increment_model"

# 执行pipeline, 是否删除过期数据(包含样本,  模型数据, 工作目录数据, True or False)
PIPELINE_CLEAN_DATA="True"

# 训练数据和模型数据的保质期限, 单位:分钟
SHELF_LIFE_TRAIN_DATA=1440
SHELF_LIFE_MODEL_DATA=720
#------------------------------------------------------#


#--------------- 模型特征&&上线相关配置 ---------------#

# 模型上线的名称 - 推送模型上线时 或者 写监控数据时 必须配置
ONLINE_DICT_NAME="weiaj_ugc_like_test"

# 模型所用特征列表, 必须放到./data/目录, 配置格式跟realtime流程中的保持一致
FEATURE_SCHEMA_FILE="schema_demo.json"
#------------------------------------------------------#


#------------------ 模型评估相关配置 ------------------#

# 预测输出的列格式, 一般情况下用以下默认的就行
# 1. 预测输出时带上用户类型: label,user_id,user_type,pctr
PRED_OUTPUT_FORMAT="label,user_id,pctr"

# 用于分维度评估的样本标签, 配置格式为: [写入tfrecord中的key]:[key对应的特征类型], 多个标签用逗号分隔
DEBUG_LABELS="user_id:int"
#------------------------------------------------------#



#------------------ 模型构建相关配置 ------------------#

# 是否开启共享embeddings(True or False)
IS_SHARE_EMB="False"

# 是否使用增量训练模式(True or False)
USE_INCREMENT_MODE="True"

# tfrecord中用作label的key名称
LABEL_KEY="click"

# tfrecord中label的数据类型, 默认为float, 可选为int
LABEL_DTYPE='int'

# 模型训练/测试所用的gpu设备号, 注意执行任务前先用nvidia-smi查看资源状态
CUDA_DEVICE_ID=5

# 激活函数，参考tf_trainer/models/__init__.py中activation可选项
ACTIVATION="relu"

# 模型种类, 参考tf_trainer/models/__init__.py中的条件判断
MODEL="default"

# 网络种类, 参考tf_trainer/networks/__init__.py中的network_factory函数
NETWORK="deepfm"

# dataset的种类, 参考tf_trainer/dataset/__init__.py中的build_dataset函数
DATASET="rec"

# attation配置, 格式为"特征名:特征名", 多个组合用逗号分割
ATTENTION_COLS="real_click_list_seq:item_id"
#------------------------------------------------------#



#-------------------- 模型训练参数配置 ----------------#

# 模型训练的任务名称
# 若多个训练任务共享同个WORKD_DIR, 可以设置不同的TASK_NAME, 方便查看对应的summary
TASK_NAME="test"

# 训练样本的迭代轮数
EPOCH_NUM=1

# 各个deep网络层的大小
DEEP_LAYERS="400,400,400"

# 学习率
LEARNING_RATE=0.001

# tfrecord数据中一条记录的样本数
PREBATCH=256

# 训练时的batch
BATCH_SIZE=1

# embedding向量的维度
EMB_DIM=8
#------------------------------------------------------#



#---------------- 模型更新校验参数配置 ----------------#
VER_THRESH_AUC=0.50                     # 模型离线auc最低值
VER_THRESH_GAUC=-1.0                    # 模型离线guac最低值
VER_THRESH_CALIBRATION_UPPER=3.0        # 模型离线calibration上限
VER_THRESH_CALIBRATION_LOWER=0.0        # 模型离线calibration下限

# 模型最低size, 单位为MB
VER_THRESH_MODEL_SIZE=0
#------------------------------------------------------#



#---------------- 模型更新邮件通知配置 ----------------#
MAIL_ADDRESS="weiaj@ushareit.com"
MAIL_SUBJECT="Test DNN Model Updating"


#END
