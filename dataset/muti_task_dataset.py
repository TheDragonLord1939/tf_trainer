import tensorflow as tf

from dataset.rec_dataset import RecDataset


class MutiTaskDataset(RecDataset):

    def __init__(self, compression_type, label_key, schema, valid_path, train_path='', batch_size=16,
                 prebatch=512, epochs=1, seq_train=False, **kwargs):
        super().__init__(compression_type, label_key, schema, valid_path, train_path,
                         batch_size, prebatch, epochs, seq_train, **kwargs)
        self.task_types = self._parse_multi_task_config(kwargs.get('multi_task_config', ''))

    @property
    def label_spec(self):
        labels = dict()
        for name in self.task_types.keys():
            labels[name] = tf.FixedLenFeature((self._prebatch,), tf.float32, [1.]*self._prebatch)
        return labels

    def _de_prebatch_parser(self, samples):
        features, labels = super()._de_prebatch_parser(samples)
        # 多目标里面的时长目标(如果有)需要log1p放缩
        if "play_duration" in labels:
            labels["play_duration"] = tf.log1p(labels["play_duration"])
        return features, labels

    def _parse_multi_task_config(self, multi_task_config):
        task_dict = dict()
        if not multi_task_config:
            return task_dict
        for task_config in multi_task_config.split(","):
            task_name, task_type = task_config.split(":")
            task_dict[task_name] = task_type
        return task_dict