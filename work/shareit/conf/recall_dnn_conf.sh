#!/usr/bin/bash


# ========== 目录相关参数 ========== #
# obs上训练数据存放目录，obs上的格式应该为${OBS_DATA_DIR}/$dt/train_data和obs上的格式应该为${OBS_DATA_DIR}/$dt/eval_data
OBS_DATA_DIR="obs://sprs-data-sg/sharezone/chengtx/sample/recall_dnn_data"

# 本地训练数据存放目录，请使用绝对路径。会创建${LOCAL_DATA_DIR}/$dt/train和${LOCAL_DATA_DIR}/$dt/train/eval目录
LOCAL_DATA_DIR="/tf/tensorflow/tf_trainer/work/shareit/data/recall_dnn"

# 本地训练的模型文件，会创建${LOCAL_MODEL_DIR}/$dt目录
LOCAL_MODEL_DIR="/tf/tensorflow/tf_trainer/work/shareit/model/recall_dnn"
LOCAL_MODEL_SCHEMA="/tf/tensorflow/tf_trainer/work/shareit/conf/recall_dnn_schema.csv"

# 增量模型目录
INCREMENTAL_MODEL_DIR="${LOCAL_MODEL_DIR}/incremental"

# 线上目录
EXPORT_MODEL_DIR="${LOCAL_MODEL_DIR}/best"													# 本地导出最好的模型的目录
ONLINE_MODEL_DIR="obs://sprs-data-sg/sharezone_model_server/online_dict/deep_models"		# 线上模型的目录
ONLINE_DICT_NAME="dnn_model_t2"
# ========== 目录相关参数 ========== #

# ========== 训练相关参数 ========== #
CUDA_DEVICE_ID="9"						# 使用哪块GPU

MODEL="youtube_dnn"						# 模型
DATASET="rec"							# 数据集类型
NETWORK="dnn"							# 网络类型

HIDDEN_UNITS="256,128,64"				# 神经元数量
OUTLAYER_UNIT="32"						# 输出层神经元数量
ACTIVATION="relu"
FIXED_EMBEDDING_DIM=8					# embedding的固定维度

LABEL_KEY="sample_ids"					# label
LABEL_TYPE="int"						# label类型
LABEL_NUM=11							# label的数量，单label设为1或者NULL

# 用于分维度评估的样本标签, 配置格式为: [写入tfrecord中的key]:[key对应的特征类型], 多个标签用逗号分隔
# 默认配置中的group_id对应user_id
DEBUG_LABELS=""

LEARNING_RATE=0.01						# 学习率
EPOCH_NUM=4								# epoch
PREBATCH_NUM=10							# prebatch
BATCH_SIZE=64							# batch size

CHECKPOINT_STEPS=10000					# 存储checkpoint的间隔
FIXED_TRAINING_STEPS=100000				# 固定训练步数，实际训练步数取决于数据量和固定训练步数的小者，开启后不运行early stop，也不做eval。0表示关闭
WARM_START_DIR=${INCREMENTAL_MODEL_DIR}	# 增量训练时使用
# ========== 训练相关参数 ========== #

# ========== 导出相关参数 ========== #
EXPORT_SOURCE="checkpoint"				# 导出源，model表示从模型导出，checkpoint表示从checkpoint导出
EXPORT_VARIABLE="nce_weights"			# 导出的变量名
# ========== 导出相关参数 ========== #
