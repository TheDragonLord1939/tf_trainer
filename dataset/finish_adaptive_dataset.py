import tensorflow as tf

from dataset.rec_dataset import RecDataset


class AdaptiveFinishDataset(RecDataset):
    def __init__(self, compression_type, label_key, schema, valid_path, train_path='', batch_size=16,
                 prebatch=512, epochs=1, seq_train=False, **kwargs):
        super().__init__(compression_type, label_key, schema, valid_path, train_path,
                         batch_size, prebatch, epochs, seq_train, **kwargs)

        self._weight_key = kwargs['weight_key']

    @property
    def label_spec(self):
        labels = dict()
        labels[self._label_key] = tf.FixedLenFeature(
            shape=(self._prebatch * self._label_num,),
            dtype=tf.float32 if self._label_dtype == 'float' else tf.int64,
            default_value=[0. if self._label_dtype == 'float' else 0] * self._prebatch
        )
        labels['ptr'] = tf.FixedLenFeature((self._prebatch,), tf.float32, [1.]*self._prebatch)
        labels['play_duration'] = tf.FixedLenFeature((self._prebatch,), tf.float32, [1.]*self._prebatch)
        return labels

    def _de_prebatch_parser(self, samples):
        features, labels = super()._de_prebatch_parser(samples)
        label = tf.cast(tf.logical_or(labels["ptr"] >= 0.8, labels['play_duration'] >= 120), tf.float32)
        # pred函数读取的是features[self._label_key], 如果不修改, 虽然不影响训练, 但是评估的metrics会有问题
        features[self._label_key] = label
        return features, label
