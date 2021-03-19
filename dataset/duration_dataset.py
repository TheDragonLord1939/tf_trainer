import tensorflow as tf

from dataset.rec_dataset import RecDataset


class DurationDataset(RecDataset):
    def _de_prebatch_parser(self, samples):
        features, labels = super()._de_prebatch_parser(samples)
        return features, tf.log1p(tf.cast(labels, dtype=tf.float32))
