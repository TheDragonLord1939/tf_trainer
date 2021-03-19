import tensorflow as tf

from dataset.rec_dataset import RecDataset


class WeightedDataset(RecDataset):
    def __init__(self, compression_type, label_key, schema, valid_path, train_path='', batch_size=16,
                 prebatch=512, epochs=1, seq_train=False, **kwargs):
        super().__init__(compression_type, label_key, schema, valid_path, train_path,
                         batch_size, prebatch, epochs, seq_train, **kwargs)

        self._weight_key = kwargs['weight_key']

    @property
    def label_spec(self):
        labels = super().label_spec
        labels[self._weight_key] = tf.FixedLenFeature([self._prebatch], tf.float32)
        return labels

    def _de_prebatch_parser(self, samples):
        features, labels = super()._de_prebatch_parser(samples)
        label = tf.cast(labels[self._label_key] >= labels[self._weight_key], tf.float32)
        weight = labels[self._label_key]
        return features, (weight, label)
