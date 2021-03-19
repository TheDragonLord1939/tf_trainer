import tensorflow as tf
from .rec_dataset import RecDataset


class RecallDataset(RecDataset):
    def __init__(self, compression_type, label_key, schema, valid_path, train_path='', batch_size=16,
                 prebatch=512, epochs=1, seq_train=False, **kwargs):
        super().__init__(compression_type, label_key, schema, valid_path, train_path,
                         batch_size, prebatch, epochs, seq_train, **kwargs)
        self._negative_rate = kwargs.get('negative_rate', 5)

    def _tfrecord_pipeline(self, file_list, is_training=False):
        dataset = super()._tfrecord_pipeline(file_list, is_training)
        if not self._is_predicting:
            dataset = dataset.map(self._batch_negative_sample(self._negative_rate))
        return dataset

    def _batch_negative_sample(self, negative_rate):
        def sampler(features, labels):
            _features = {key: [value] for key, value in features.items()}
            _labels = [labels]
            indices = tf.constant(list(range(self._prebatch * self._batch_size)))
            if negative_rate > 0:
                for i in range(negative_rate):
                    _indices = tf.random.shuffle(indices, seed=i*2)
                    for key in features.keys():
                        if self.tower_index.get(key, 0) == 1:
                            _features[key].append(tf.gather(features[key], _indices))
                        else:
                            _features[key].append(features[key])
                    _labels.append(tf.zeros_like(labels, dtype=tf.float32))
            __features = {key: tf.concat(value, 0) for key, value in _features.items()}
            __labels = tf.concat(_labels, 0)
            return __features, __labels
        return sampler
