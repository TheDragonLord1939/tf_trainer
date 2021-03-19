import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Zeros, glorot_uniform, uniform
from tensorflow.python.keras.layers import Activation, BatchNormalization, Layer, Dropout
from tensorflow.python.keras.regularizers import l2


class SafeKEmbedding(tf.keras.layers.Embedding):
    def compute_mask(self, inputs, mask=None):
        oov_mask = tf.less(inputs, self.input_dim)
        if not self.mask_zero:
            return oov_mask
        return tf.logical_and(oov_mask, tf.not_equal(inputs, 0))

    def call(self, inputs):
        out = super().call(inputs)
        mask = tf.expand_dims(tf.cast(self.compute_mask(inputs), K.dtype(out)), -1)
        return out * mask


class CategorizeLayer(Layer):
    """
    make numerical feature categorical
    Input Tensor:
     inputs: numerical tensor with type tf.float32 and shape batchsize x 1
    Output Tensor:
     outputs: categorical tensor with type tf.int64 and shape batchsize x 1
    """

    def __init__(self, voc_size, ratio=100):
        self.voc_size = voc_size
        self.ratio = ratio
        super().__init__()

    def call(self, inputs, **kwargs):
        outputs = inputs * self.ratio
        outputs = tf.cast(outputs, tf.int64)
        return tf.minimum(outputs, self.voc_size - 1)

    def get_config(self):
        config = {"voc_size": self.voc_size, "ratio": self.ratio}
        base_config = super().get_config()
        config.update(base_config)
        return config


class CategorizeEmbeddingLayer(Layer):
    """
    make numerical feature categorical
    Input Tensor:
     inputs: numerical tensor with type tf.float32 and shape batchsize x 1
    Output Tensor:
     outputs: categorical tensor with type tf.int32 and shape batchsize x 1 x dim
    """

    def __init__(self, dim, **kwargs):
        self.dim = dim
        self.kernel = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = tf.get_variable(name='embedding_{}'.format(self.name),
                                      shape=(1, self.dim),
                                      initializer=uniform,
                                      trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = tf.matmul(inputs, self.kernel)
        outputs = tf.expand_dims(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1, self.dim

    def get_config(self):
        config = {"dim": self.dim}
        base_config = super().get_config()
        config.update(base_config)
        return config


def inner_product(a: tf.Tensor, b: tf.Tensor):
    return tf.reduce_sum(tf.multiply(a, b), -1)


def features_corss(a: tf.Tensor, b: tf.Tensor, weights: tf.Tensor = 1.0):
    """
    :param a: tensor with shape batch_size x 1
    :param b: tensor with shape batch_size x n
    :param weights: same shape with b
    :return: a crossed tensor with shape batch_size x 1
    """
    cross = tf.cast(b == a, tf.float32)
    if weights != 1.0:
        weights = weights / (tf.reduce_sum(weights, axis=-1, keepdims=True) + 1e-15)
    return tf.reduce_max(cross * weights, -1, keepdims=True)


def _init_embedding_weights(layer_name, voc_size, emb_size, name_scope):
    """
    initialize embedding_weights, where
    id 0 is reserved for UNK, and its embedding fix to all zeros
    """
    with tf.variable_scope('{}_{}'.format(name_scope, layer_name)):
        return tf.get_variable('embeddings', [voc_size, emb_size])


def _safe_ids_for_emb(ids, voc_size):
    """
    if id >= voc_size, then it set to 0 which means UNK, and the embedding
    should be all zeros.
    """
    return tf.where(tf.less(ids, voc_size), ids, tf.zeros_like(ids))


def embedding(ids, layer_name, voc_size, emb_size, name_scope='embedding'):
    """
    input ids shape: B x 1
    output shape: B x E
    """
    emb = _init_embedding_weights(layer_name, voc_size, emb_size, name_scope)
    out = tf.nn.embedding_lookup(emb, ids, name=layer_name)
    out = tf.reshape(out, [-1, emb_size])
    return out


def _safe_ids_weights_for_sparse_emb(ids, voc_size, weights=None):
    """
    if ids.values >= voc_size, it set to 0, its weight also set to 0
    """
    mask = tf.less(ids.values, voc_size)
    zeros = tf.zeros_like(ids.values)
    weights_value = weights.values if weights else tf.ones_like(ids.values)
    safe_ids = tf.SparseTensor(
        indices=ids.indices,
        values=tf.where(mask, ids.values, zeros),
        dense_shape=ids.dense_shape)
    safe_weights = tf.SparseTensor(
        indices=ids.indices,
        values=tf.where(mask, weights_value, zeros),
        dense_shape=ids.dense_shape)
    return safe_ids, safe_weights


def sparse_embedding(ids, layer_name, voc_size, emb_size, name_scope='embedding', combiner='mean'):
    """
    input ids shape: B x V
    output shape: B x E
    """
    emb = _init_embedding_weights(layer_name, voc_size, emb_size, name_scope)
    out = tf.nn.embedding_lookup_sparse(emb, ids, None, combiner=combiner, name=layer_name)
    out = tf.reshape(out, (-1, 1, emb_size))
    return out


class DeepLayer(Layer):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super().__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel_{}'.format(i),
                                        shape=(hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_uniform(seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias_{}'.format(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [Dropout(self.dropout_rate, seed=self.seed + i) for i in range(len(self.hidden_units))]

        activations = {
            "leaky_relu": tf.keras.layers.LeakyReLU(),
            "swish": tf.nn.swish
        }
        self.activation_layer = activations[self.activation] if self.activation in activations else Activation(self.activation)

        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activation_layer(fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate, 'seed': self.seed}
        base_config = super().get_config()
        config.update(base_config)
        return config


class LocalAttentionUnit(Layer):
    """The LocalActivationUnit used in DIN with which the representation of
    user interests varies adaptively given different candidate items.
      Input shape
        - A list of two 3D tensor with shape:  ``(batch_size, 1, embedding_size)`` and ``(batch_size, T, embedding_size)``
      Output shape
        - 3D tensor with shape: ``(batch_size, T, 1)``.
      Arguments
        - **activation**: Activation function to use in attention net.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix of attention net.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout in attention net.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not in attention net.
        - **seed**: A Python integer to use as random seed.
      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of
        the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.]
        (https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, activation='sigmoid', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024,
                 **kwargs):
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        super(LocalAttentionUnit, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `LocalActivationUnit` layer should be called on a list of 2 inputs')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError("Unexpected inputs dimensions {} and {}, \
                             expect to be 3 dimensions".format(len(input_shape[0]), len(input_shape[1])))

        if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1:
            raise ValueError('A `LocalActivationUnit` layer requires '
                             'inputs of a two inputs with shape (None,1,embedding_size) and (None,T,embedding_size)'
                             'Got different shapes: {}'.format(input_shape))
        hidden_units = [2 * input_shape[0][-1].value, input_shape[0][-1].value, 1]
        self.dnn = DeepLayer(hidden_units, self.activation, self.l2_reg,
                             self.dropout_rate, self.use_bn, seed=self.seed, name='{}_net'.format(self.name))
        super(LocalAttentionUnit, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):
        query, keys = inputs
        queries = query * tf.ones_like(keys)
        att_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
        att_out = self.dnn(att_input, training=training)
        return att_out

    def compute_output_shape(self, input_shape):
        return input_shape[1][:2] + (1,)

    def get_config(self, ):
        config = {'activation': self.activation, 'l2_reg': self.l2_reg,
                  'dropout_rate': self.dropout_rate, 'use_bn': self.use_bn, 'seed': self.seed}
        base_config = super(LocalAttentionUnit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
