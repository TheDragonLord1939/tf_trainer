import tensorflow as tf
from networks.base_network import Network
from layers import DeepLayer


class Dnn(Network):

    def _build(self):
        self.deep = DeepLayer(
            [int(i) for i in self._hidden.split(',')],
            activation=self._flags.activation,
            l2_reg=self._flags.l2,
            dropout_rate=self._flags.dropout,
            use_bn=self._flags.use_bn
        )
        super()._build()

    def call(self, dense, embeddings, is_training):
        with tf.name_scope(name=self._name):
            if embeddings == None:
                x = tf.concat(dense, -1)
            else:
                x = tf.concat(dense + [tf.squeeze(emb, [1]) for emb in embeddings], -1)
            deep_out = self.deep(x)
            return deep_out
