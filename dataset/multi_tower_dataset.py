import tensorflow as tf

from dataset.rec_dataset import RecDataset


"""
多塔数据集的数据格式为：
（1）左塔特征，tower_index = 0
（2）右塔特征，tower_index = 1
（3）样本值，使用label来表示
多塔数据和rec数据集的差异在于：
    rec数据集中，只有label是sample_num个，而多塔数据中，右塔特征和label都是sample_num个
    因此，构建样本时，所有的右塔特征，都是数组类型，相当于按照sample_num去做batch。数据集在处理时，使用pre_batch * sample_num去做de_prebatch
"""

class MultiTowerDataset(RecDataset):
    def __init__(self, compression_type, label_key, schema, valid_path, train_path='', batch_size=16,
                 prebatch=512, epochs=1, seq_train=False, **kwargs):
        super().__init__(compression_type, label_key, schema, valid_path, train_path,
                         batch_size, prebatch, epochs, seq_train, **kwargs)

    def _make_tfrecord_parser(self, record):
        features_spec = self.feature_spec
        label_spec    = self.label_spec

        for feature_name in features_spec:
            tower_index = self.tower_index.get(feature_name, 0)
            if self._label_num > 1 and tower_index == 1:                    # 右塔是多塔时，特征全部都是数组类型的，需要额外处理
                if feature_name in self.categorical_list:                   # Int类型
                    features_spec[feature_name] = tf.io.FixedLenFeature([self._prebatch * self._label_num], tf.int64, [0] * self._prebatch * self._label_num)
                if feature_name in self.sequence_list:
                    features_spec[feature_name] = tf.io.FixedLenFeature([self._prebatch * self._label_num * self.sequence_length[feature_name]],
                                                                        tf.float32 if feature_name+'_weight' in self.weight_sequence_list else tf.int64,
                                                                        [0] * self._prebatch * self._label_num * self.sequence_length[feature_name])
                if feature_name in self.vector_list:
                    features_spec[feature_name] = tf.io.FixedLenFeature([self._prebatch * self._label_num * self.vec_dim[feature_name]], tf.float32,
                                                                        [0.] * self._prebatch * self._label_num * self.vec_dim[feature_name])
                if feature_name in self.numerical_list:                             # Double类型
                    features_spec[feature_name] = tf.io.FixedLenFeature([self._prebatch * self._label_num], tf.float32, [0.] * self._prebatch * self._label_num)
   
        features_spec.update(label_spec)

        tf.logging.debug("%s parse_single_example: %r" % (self.__class__.__name__, features_spec))
        features = tf.parse_single_example(record, features_spec)

        return features

