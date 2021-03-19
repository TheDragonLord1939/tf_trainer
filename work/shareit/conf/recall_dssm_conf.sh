#!/usr/bin/bash


# ========== 目录相关参数 ========== #
# obs上训练数据存放目录，obs上的格式应该为${OBS_DATA_DIR}/$dt/train_data和obs上的格式应该为${OBS_DATA_DIR}/$dt/eval_data
OBS_DATA_DIR="obs://sprs-data-sg/sharezone/recall/model/funu/dssm"
OBS_PRED_DATA_DIR="obs://sprs-data-sg/sharezone/recall/model/funu/dssm"
# 本地训练数据存放目录，请使用绝对路径。会创建${LOCAL_DATA_DIR}/$dt/train和${LOCAL_DATA_DIR}/$dt/train/eval目录
LOCAL_DATA_DIR="/tf/tensorflow/tf_trainer/work/shareit/data/dssm"

# 本地训练的模型文件，会创建${LOCAL_MODEL_DIR}/$dt目录
LOCAL_MODEL_DIR="/tf/tensorflow/tf_trainer/work/shareit/model/dssm"
LOCAL_MODEL_SCHEMA="/tf/tensorflow/tf_trainer/work/shareit/conf/recall_dssm_schema.csv"

# run.sh所在的相对目录 - 以tf_trainer为根目录
RUN_DIR="/tf/tensorflow/tf_trainer/work/shareit"

# 增量模型目录
INCREMENTAL_MODEL_DIR="${LOCAL_MODEL_DIR}/incremental"

# 线上目录
EXPORT_MODEL_DIR="${LOCAL_MODEL_DIR}/best"													# 本地导出最好的模型的目录
#ONLINE_MODEL_DIR="obs://sprs-data-sg/sharezone_model_server/online_dict/deep_models"		# 线上模型的目录
#ONLINE_DICT_NAME="scene3_dnn_test3_model"
# ========== 目录相关参数 ========== #

# ========== 训练相关参数 ========== #
CUDA_DEVICE_ID="4"						# 使用哪块GPU

MODEL="dssm"							# 模型
DATASET="rec"							# 数据集类型
NETWORK="dnn"							# 网络类型

# 当数据集为全正例时可开启batch内负采样
NEGATIVE_RATE=0
HIDDEN_UNITS="256,128,64"				# 神经元数量
OUTLAYER_UNIT="32"						# 输出层神经元数量
ACTIVATION="relu"
FIXED_EMBEDDING_DIM=8					# embedding的固定维度

LABEL_KEY="is_issue"					# label
LABEL_TYPE="float"						# label类型
LABEL_NUM=1								# label的数量，单label设为1或者NULL

# 用于分维度评估的样本标签, 配置格式为: [写入tfrecord中的key]:[key对应的特征类型], 多个标签用逗号分隔
# 默认配置中的group_id对应user_id
DEBUG_LABELS="group_id:int"           # predict变量
DEBUG_LABELS_EMB="item_lang_string:str,raw_item_id:str" # embedding变量

# 预测输出的列格式, 一般情况下用以下默认的就行
# 1. 预测输出时带上用户类型: label,user_id,user_type,pctr
PRED_OUTPUT_FORMAT="label,user_id,pctr"

LEARNING_RATE=0.01						# 学习率
EPOCH_NUM=1								# epoch
PREBATCH_NUM=1024							# prebatch
BATCH_SIZE=8							# batch size

CHECKPOINT_STEPS=1000					# 存储checkpoint的间隔
FIXED_TRAINING_STEPS=0					# 固定训练步数，实际训练步数取决于数据量和固定训练步数的小者，开启后不运行early stop，也不做eval。0表示关闭
WARM_START_DIR=${INCREMENTAL_MODEL_DIR}	# 增量训练时使用
# ========== 训练相关参数 ========== #

# ========== 导出相关参数 ========== #
EXPORT_VARIABLE="item"			# 导出的变量名
# ========== 导出相关参数 ========== #
