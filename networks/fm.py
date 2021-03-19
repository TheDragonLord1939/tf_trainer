import tensorflow as tf

from layers import fm_net
from networks.base_network import Network


class Fm(Network):

    def call(self, dense, embeddings, is_training):
        with tf.name_scope(name=self._name):
            fm_in = tf.concat(embeddings, 1)
            fm_out = fm_net(fm_in)
            return fm_out
