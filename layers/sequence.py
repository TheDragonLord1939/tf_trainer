import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from layers import LocalAttentionUnit


class SequencePoolingLayer(Layer):
    """The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length sequence feature/multi-value feature.
      Input shape
        - A list of two  tensor [seq_value,seq_len]
        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``
        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.
      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.
      Arguments
        - **mode**:str.Pooling operation to be used,can be sum, mean or sqrtn.
        - **supports_masking**:If True,the input need to support masking.
    """

    def __init__(self, mode='mean', supports_masking=False, **kwargs):

        if mode not in ('sum', 'mean', 'sqrtn'):
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        self.eps = 1e-8
        super(SequencePoolingLayer, self).__init__(**kwargs)

        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = input_shape[0][1].value
        super(SequencePoolingLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError("When supports_masking=True,input must support masking")
            uiseq_embed_list = seq_value_len_list
            mask = tf.cast(mask, tf.float32)
            user_behavior_length = tf.reduce_sum(mask, axis=-1, keep_dims=True)
            mask = tf.expand_dims(mask, axis=2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list

            mask = tf.sequence_mask(user_behavior_length, self.seq_len_max, dtype=tf.float32)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = uiseq_embed_list.shape[-1]

        mask = tf.tile(mask, [1, 1, embedding_size])

        uiseq_embed_list *= mask
        hist = tf.reduce_sum(uiseq_embed_list, 1, keep_dims=False)

        if self.mode == 'mean':
            hist = tf.div(hist, user_behavior_length + self.eps)
        elif self.mode == 'sqrtn':
            hist = tf.div(hist, tf.sqrt(user_behavior_length) + self.eps)
        return tf.expand_dims(hist, axis=1)

    def compute_output_shape(self, input_shape):
        if self.supports_masking:
            return None, 1, input_shape[-1]
        else:
            return None, 1, input_shape[0][-1]

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self, ):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(SequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SequenceWeightedPoolingLayer(Layer):
    """The SequencePoolingLayer is used to apply weighted pooling on sequence feature/multi-value feature.
      Input shape
        - sequence is a 3D tensor with shape: ``(batch_size, T, embedding_size)``
        - weights is a 2D tensor with shape : ``(batch_size, T)``.
      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.
    """

    def __init__(self, **kwargs):

        self.eps = 1e-8
        super().__init__(**kwargs)

    def call(self, sequence, weights=None, **kwargs):
        if weights is None:
            weights = tf.ones(sequence.get_shape[:2], dtype=tf.float32)
        sequence *= tf.expand_dims(weights, axis=-1)
        hist = tf.reduce_sum(sequence, 1, keep_dims=True)
        norm = tf.expand_dims(tf.reduce_sum(weights, -1, keep_dims=True), 1)
        return tf.div(hist, norm + self.eps)

    def compute_output_shape(self, input_shape):
        return None, 1, input_shape[-1]

    def compute_mask(self, inputs, mask=None):
        return None


class DinSequencePoolingLayer(Layer):
    """The Attentional sequence pooling operation used in DIN.
      Input shape
        - A list of three tensor: [query,keys,keys_length]
        - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``
        - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``
        - mask is a 2D tensor with shape:   ``(batch_size, 1)``
      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.
      Arguments
        - **att_activation**: Activation function to use in attention net.
        - **weight_normalization**: bool.Whether normalize the attention score of local activation unit.
        - **supports_masking**:If True,the input need to support masking.
      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """
    def __init__(self, att_activation='sigmoid', weight_norm=False, return_score=False, **kwargs):
        self.weight_norm = weight_norm
        self.att_activation = att_activation
        self.return_score = return_score
        self.minus_inf = -2 ** 32 + 1
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        self.local_att = LocalAttentionUnit(activation=self.att_activation, l2_reg=0, dropout_rate=0,
                                            use_bn=False, name='attention_{}'.format(self.name))

    def call(self, inputs, mask=None, training=None, **kwargs):
        if mask is None:
            raise ValueError('input masking must be supplied')
        attention_scores = self.local_att(inputs, trainable=training)
        _, keys = inputs
        padding = tf.ones_like(attention_scores) * self.minus_inf if self.weight_norm \
            else tf.zeros_like(attention_scores)
        keys_mask = tf.expand_dims(mask, axis=-1)
        outputs = tf.where(keys_mask, attention_scores, padding)
        if self.weight_norm:
            outputs = tf.nn.softmax(outputs, axis=1)
        if not self.return_score:
            outputs = tf.reduce_sum(keys * outputs, axis=1, keep_dims=True)
        return outputs

    def get_config(self, ):
        config = {'att_hidden_units': self.att_hidden_units, 'att_activation': self.att_activation,
                  'weight_normalization': self.weight_normalization, 'return_score': self.return_score,
                  'supports_masking': self.supports_masking}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
