#!/bin/bash

#---------- 数据存放和获取相关配置 ----------#

# tfrecord数据在obs上存放的目录
BASE_CDN_DATA_DIR="obs://sprs-data-sg/sharezone/zhaoyq/likeit_ugc_id/"

# 线上模型在obs上存放的目录
BASE_ONLINE_DICT_DIR="obs://sprs-data-sg/sharezone_model_server/online_dict/deep_models"

# 线上模型在本地存放的目录
BASE_ONLINE_DICT_LOCAL_DIR="/tf/tensorflow//zhaoyq/online_dict/single_model/is_effective_4_e1/"

# 执行run.sh相关任务的工作目录, 用于存放模型、日志等
WORK_DIR="/tf/tensorflow//zhaoyq/single_model/is_effective_4_e1/"

# run.sh所在的相对目录 - 以tf_trainer为根目录
RUN_DIR="./work/weshare/"

# 增量模型所在的本地目录, 若模型非增量设置为空即可
INCREMENT_MODEL_DIR="${WORK_DIR}increment_model"

# 增量模型云端备份目录, 设置为空时不备份到云端
INCREMENT_MODEL_CLOUD_DIR=""


user_cross_feature="user_id,user_level,real_play_list_size,real_like_list_size,real_share_list_size,real_download_list_size,real_follow_list_size,real_follow_author_list_size,real_finish_list_size,real_play_list,real_finish_list,user_real_labels_pos,user_real_labels_neg,req_province,user_real_labels2_pos,user_real_labels2_neg,is_new_user,user_short_labels"
item_cross_feature="tags,duration,source_user,final_label1,final_label2,effective_rate,finish_rate,con_effective_rate,con_finish_rate,hashtags,bgm_id,author_id,publish_dispersed_day,countries,langs"

user_features_list_gate_controled="user_id,real_play_list,real_finish_list,user_real_labels_pos,user_real_labels_neg,user_real_labels2_pos,user_real_labels2_neg,user_short_labels"

gate_input_feature_list="user_level,real_play_list_size,real_like_list_size,real_share_list_size,real_download_list_size,real_follow_list_size,real_follow_author_list_size,real_finish_list_size"
# ssd - 表示数据下载到ssd上,  mhd - 表示数据下载到mhd上
WORK_ON_SSD_MHD="ssd"
MHD_PATH="/data"        # 机械盘根目录
SSD_PATH="/data2"       # 固态盘根目录

# 执行pipeline1 or pipeline2时, 是否删除过期数据(包含样本,  模型数据, 工作目录数据, True or False)
PIPELINE_CLEAN_DATA="False"
#--------------------------------------------#



#---------- 模型特征&&上线相关配置 ----------#

# 模型上线的名称 - 推送模型上线时 或者 写监控数据时 必须配置
ONLINE_DICT_NAME="scene3_dnn_test2_model"

# 在线预测时模型不需要的特征列表, 多个特征用逗号分隔
ONLINE_BLACK_FEATURES=""

# 模型所用特征列表, 必须放到./data/目录
FEATURE_MAP_NAME="ugc_finish_v1.0.csv"

# 生成model_server所用特征词典的脚本
# 非增量模型使用gen_feature_map_total.py, 增量模型使用gen_feature_map.py
FEATURE_INDEX_SCRIPT="gen_feature_map.py"

# 是否使用hash词典模式产出模型
DNN_WITH_HASH="False"
#--------------------------------------------#



#---------- 模型评估相关配置 ----------#

# 是否使用后验方式来验证 (昨天的模型来验证今日的数据, True or False)
IS_POSTERIOR_VALID="False"

# 预测输出的列格式, 一般情况下用以下默认的就行
# 1. 预测输出时带上用户类型: label,user_id,user_type,pctr
PRED_OUTPUT_FORMAT="label,user_id,pctr"

# 用于分维度评估的样本标签, 配置格式为: [写入tfrecord中的key]:[key对应的特征类型], 多个标签用逗号分隔
# 默认配置中的group_id对应user_id
DEBUG_LABELS="group_id:int"
#--------------------------------------#



#---------- 模型构建相关配置 ----------#

# 是否开启共享embeddings(True or False)
IS_SHARE_EMB="True"

# 是否使用增量训练模式(True or False)
USE_INCREMENT_MODE="True"

# tfrecord中用作label的key名称
LABEL_KEY="is_effective_4"

# tfrecord中label的数据类型, 默认为float, 可选为int
LABEL_DTYPE='int'

# 模型训练/测试所用的gpu设备号, 注意执行任务前先用nvidia-smi查看资源状态
CUDA_DEVICE_ID=6

# 激活函数，参考tf_trainer/models/__init__.py中activation可选项
ACTIVATION="leaky_relu"

# 模型种类, 参考tf_trainer/models/__init__.py中的条件判断
MODEL="deepfm_diff_lr_user_id_item_bias"

# 网络种类, 参考tf_trainer/networks/__init__.py中的network_factory函数
NETWORK="deepfm_bias_gate_cross"

# dataset的种类, 参考tf_trainer/dataset/__init__.py中的build_dataset函数
DATASET="rec"

# attation配置, 格式为"特征名:特征名", 多个组合用逗号分割
ATTENTION_COLS=""
#--------------------------------------#



#---------- 模型训练参数配置 ----------#

# 模型训练的任务名称
# 若多个训练任务共享同个WORKD_DIR, 可以设置不同的TASK_NAME, 方便查看对应的summary
TASK_NAME="base"

# 训练样本的迭代轮数
EPOCH_NUM=1

# 各个deep网络层的大小
DEEP_LAYERS="400,400,400"

# 学习率
LEARNING_RATE=0.001

# tfrecord数据中一条记录的样本数
PREBATCH=128

# 训练时的batch
BATCH_SIZE=32

# embedding向量的维度
EMB_DIM=8
#--------------------------------------#



#---------- 模型更新校验参数配置 ----------#
VER_THRESH_AUC=0.70                     # 模型离线auc最低值
VER_THRESH_GAUC=-1.0                    # 模型离线guac最低值
VER_THRESH_CALIBRATION_UPPER=1.2        # 模型离线calibration上限
VER_THRESH_CALIBRATION_LOWER=0.8        # 模型离线calibration下限

# 模型最低size, 单位为MB
VER_THRESH_MODEL_SIZE=500
#------------------------------------------#



#---------- 模型更新邮件通知配置 ----------#
MAIL_ADDRESS="zhaoyq@ushareit.com,zhaoyq@ushareit.com"
MAIL_SUBJECT="Feed DNN Model Updating"
#------------------------------------------#
