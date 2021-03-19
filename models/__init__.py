import tensorflow as tf


# model config
tf.flags.DEFINE_boolean('distribute', False, 'if use multiple gpus')
tf.flags.DEFINE_boolean('freeze_embeddings', False, 'whether to freeze embeddings')
tf.flags.DEFINE_boolean('seq_train', False, 'whether to train by seq')
tf.flags.DEFINE_enum('compression_type', '', ['', 'GZIP', 'ZLIB'], 'compression type of tfrecords')
tf.flags.DEFINE_enum('label_dtype', 'float', ['float', 'int'], 'dtype of label, default float')
tf.flags.DEFINE_enum('mode', 'train', ['train', 'test', 'predict'], 'run mode of the model')
tf.flags.DEFINE_enum('warm_start_mode', 'emb', ['all', 'emb'], 'emb: keep embeddings only, all: keep all')
tf.flags.DEFINE_float('dropout', 0., 'dropout rate for dnn')
tf.flags.DEFINE_float('l2', 0., 'l2 regularization')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate for optimizer')
tf.flags.DEFINE_integer('batch_size', 8, 'emb factor for calc emb dim')
tf.flags.DEFINE_integer('checkpoints_steps', 1000, 'steps for saving checkpoint')
tf.flags.DEFINE_integer('emb_factor', 6, 'emb factor for calc emb dim')
tf.flags.DEFINE_integer('epochs', 2, 'epochs for training')
tf.flags.DEFINE_integer('eval_throttle', 120, 'throttle between evals')
tf.flags.DEFINE_integer('fixed_emb_dim', None, 'emb dim fixed')
tf.flags.DEFINE_integer('fixed_training_steps', None, 'fixed training step without eval if provided')
tf.flags.DEFINE_integer('log_step_count_steps', 1000, 'logging step count')
tf.flags.DEFINE_integer('num_experts', 8, 'num of experts')
tf.flags.DEFINE_integer('negative_rate', 5, 'negative rate for batch negative sampling')
tf.flags.DEFINE_integer('patient', 8000, 'epochs for training')
tf.flags.DEFINE_integer('prebatch', 1, 'size of prebatch')
tf.flags.DEFINE_integer('save_summary_steps', 200, 'steps for saving summary')
tf.flags.DEFINE_integer('vector_dim', None, 'vector dim of dssm and youtube dnn vector')
tf.flags.DEFINE_string('checkpoint_dir', '/tmp/tf_checkpoint/', 'path to export checkpoint')
tf.flags.DEFINE_string('dataset', 'rec', 'type of dataset')
tf.flags.DEFINE_string('duration_thresholds', '', 'duration split thresholds, joined by ","')
tf.flags.DEFINE_string('feature_influence', None, 'feature name to evaluate feature influence')
tf.flags.DEFINE_string('label_key', 'label', 'key of label, e.g. label')
tf.flags.DEFINE_string('model', 'default', 'model version to run')
tf.flags.DEFINE_string('debug_labels', '', "labels parsed from tfrecord for evaluation, e.g 'group_id:int'")
tf.flags.DEFINE_boolean('predict_with_emb', False, 'True: ouput item embedding, False: output predict txt')
tf.flags.DEFINE_boolean('predict_with_user', False, 'True: ouput user embedding, False: output predict txt')
tf.flags.DEFINE_string('network', '', 'networks to the model, e.g. lr')
tf.flags.DEFINE_string('schema', '', 'feature schema file path')
tf.flags.DEFINE_string('train_path', '', 'train set file path or dir')
tf.flags.DEFINE_string('valid_path', '', 'valid set file path or dir')
tf.flags.DEFINE_string('pred_path',  '', 'predictions output path')
tf.flags.DEFINE_string('warm_start_dir', '', 'path to warm up checkpoint dir')
tf.flags.DEFINE_string('weight_key', 'label_weight', 'key of label weight, e.g. duration')
# network config
tf.flags.DEFINE_boolean('emb_increase', False, 'whether to increase embedding')
tf.flags.DEFINE_boolean('norm_attention', False, 'whether to norm attention out in DIN')
tf.flags.DEFINE_boolean('res_deep', False, 'whether to use resnet for deep layer')
tf.flags.DEFINE_boolean('safe_embedding', True, 'whether to use safe embedding')
tf.flags.DEFINE_boolean('split_half', False, 'whether to split half in cin')
tf.flags.DEFINE_boolean('use_bn', False, 'whether to use bn in dnn')
tf.flags.DEFINE_boolean('use_sequence_weight', False, 'whether to use sequence weight')
tf.flags.DEFINE_enum('activation', 'relu', ['relu', 'leaky_relu', 'swish', 'elu', 'selu', 'tanh', 'sigmoid', 'linear'],
                     'activation function for deep layer')
tf.flags.DEFINE_enum('combiner', 'mean', ['mean', 'sum', 'sqrtn'], 'combiner of sparse emb lookup')
tf.flags.DEFINE_enum('categorize_numerical', 'False', ['False', 'embedding', 'bucket'], 'categorize numerical type')
tf.flags.DEFINE_integer('cross_depth', 3, 'depth for cross layer')
tf.flags.DEFINE_string('attention_cols', '', 'attentions col pairs, e.g. keys1:query1,keys2:query2')
tf.flags.DEFINE_string('cin_layers', '128,128', 'layers of cin')
tf.flags.DEFINE_string('hidden', '512,256,128', 'hidden size of deep layer')
tf.flags.DEFINE_integer('label_num', 1, 'the num of labels')
tf.flags.DEFINE_string('pal_bias', '16,16', 'hidden size of deep layer')
tf.flags.DEFINE_string('multi_task_config', 'is_like:classification,play_duration:regression', 'multi task config')
tf.flags.DEFINE_string('pal_submodel', 'dnn', 'hidden size of deep layer')

tf.flags.DEFINE_string('gate_input_feature_list', '', 'the features for gate input')
tf.flags.DEFINE_string('user_features_list_gate_controled', '', 'the features for gate to control')
tf.flags.DEFINE_string('user_cross_feature', '', 'user features for cross ')
tf.flags.DEFINE_string('item_cross_feature', '', 'item features for cross')


_model_ver = tf.flags.FLAGS.model

if _model_ver == 'weighted':
    from models.weighted_model import WeightedModel as Model
elif _model_ver == 'mmoe':
    from models.mmoe_model import MmoeModel as Model
elif _model_ver == 'mmoe_adaptation':
    from models.mmoe_adaptation_model import MmoeAdaptationModel as Model
elif _model_ver == 'dssm':
    from models.dssm_model import DssmModel as Model
elif _model_ver == 'multi_tower':
    from models.multi_tower_model import MultiTowerModel as Model
elif _model_ver == 'youtube_dnn':
    from models.youtube_dnn_model import YoutubeDNNModel as Model
elif _model_ver == 'regressor':
    from models.regressor_model import RegressorModel as Model
elif _model_ver == 'default':
    from models.base_model import BaseModel as Model
elif _model_ver == 'fm_recall':
    from models.fm_recall_model import FmRecallModel as Model
elif _model_ver == 'pal':
    from models.pal_model import PalModel as Model
elif _model_ver == 'deepfm_diff_lr_user_id_item_bias':
    from models.deepfm_diff_lr_user_id_item_bias import Deepfm_diff_lr_user_id_item_bias as Model
else:
    raise ValueError("model {} is unrecognized")

model = Model()
