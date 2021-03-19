import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, Activation
from tensorflow.python.keras.initializers import Zeros, TruncatedNormal, random_normal
from tensorflow.python.keras.regularizers import l2


class CrossNet(Layer):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **layer_num**: Positive integer, the cross layer number
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17.
        ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
    """

    def __init__(self, layer_num=2, l2_reg=0, seed=1024, **kwargs):
        self.layer_num = layer_num
        self.l2_reg = l2_reg
        self.seed = seed
        super().__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

        dim = input_shape[-1].value
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(dim, 1),
                                        initializer=random_normal(stddev=0.0005, seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True)
                        for i in range(self.layer_num)]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(dim,),
                                     initializer=Zeros(),
                                     trainable=True)
                     for i in range(self.layer_num)]
        # Be sure to call this somewhere!
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        x_0 = inputs
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = tf.matmul(x_l, self.kernels[i])
            dot_ = x_0 * xl_w
            x_l += tf.nn.bias_add(dot_, self.bias[i])
        return x_l

    def get_config(self, ):

        config = {'layer_num': self.layer_num,
                  'l2_reg': self.l2_reg, 'seed': self.seed}
        base_config = super().get_config()
        config.update(base_config)
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


def bi_interaction_layer(embeddings):
    square_of_sum = tf.square(tf.reduce_sum(embeddings, axis=1, keepdims=True))
    sum_of_square = tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=True)
    outputs = 0.5 * (square_of_sum - sum_of_square)
    return tf.reshape(outputs, (-1, outputs.shape.dims[-1]))


def fm_net(embeddings):
    first_order = tf.reduce_mean(embeddings, -1)  # B x n
    square_of_sum = tf.square(tf.reduce_sum(embeddings, axis=1))
    sum_of_square = tf.reduce_sum(tf.square(embeddings), axis=1)
    second_order = 0.5 * (square_of_sum - sum_of_square)  # B x D
    return tf.concat((first_order, second_order), -1)


def mvm(embeddings, factor_size):
    num_features = int(embeddings.shape.dims[1])
    bias = tf.get_variable("padding_bias", (num_features, factor_size), initializer=TruncatedNormal(stddev=0.02))
    all_order = tf.add(embeddings, bias)
    out = all_order[:, 0, :]  # B x 1 x factor_size
    for i in range(1, num_features):
        out = tf.multiply(out, all_order[:, i, :])
    out = tf.reshape(out, shape=[-1, factor_size])
    return out


class CIN(Layer):
    """Compressed Interaction Network used in xDeepFM.This implemention is
    adapted from code that the author of the paper published on https://github.com/Leavingseason/xDeepFM.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, featuremap_num)`` ``featuremap_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]`` if ``split_half=True``,else  ``sum(layer_size)`` .

      Arguments
        - **layer_size** : list of int.Feature maps in each layer.

        - **activation** : activation function used on feature maps.

        - **split_half** : bool.if set to False, half of the feature maps in each hidden will connect to output unit.

        - **seed** : A Python integer to use as random seed.

      References
        - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(self, layer_size=(128, 128), activation='relu', split_half=True, l2_reg=1e-5, **kwargs):
        if len(layer_size) == 0:
            raise ValueError(
                "layer_size must be a list(tuple) of length greater than 1")
        self.layer_size = layer_size
        self.split_half = split_half
        self.activation = activation
        self.l2_reg = l2_reg
        self.filters = list()
        self.bias = list()
        super(CIN, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions {}, expect to be 3 dimensions".format(len(input_shape)))

        self.field_nums = [input_shape[1].value]
        for i, size in enumerate(self.layer_size):
            self.filters.append(self.add_weight(name='filter' + str(i),
                                                shape=[1, self.field_nums[-1] * self.field_nums[0], size],
                                                dtype=tf.float32,
                                                regularizer=l2(self.l2_reg)))
            self.bias.append(self.add_weight(name='bias' + str(i), shape=[size], dtype=tf.float32,
                                             initializer=Zeros()))
            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True")

                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)
        self.activation_layers = [Activation(self.activation) for _ in self.layer_size]
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions {}, expect to be 3 dimensions".format(K.ndim(inputs)))
        dim = inputs.get_shape()[-1].value
        hidden_nn_layers = [inputs]
        final_result = list()
        split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)
        for idx, layer_size in enumerate(self.layer_size):
            split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)
            dot_result_m = tf.matmul(
                split_tensor0, split_tensor, transpose_b=True)
            dot_result_o = tf.reshape(
                dot_result_m, shape=[dim, -1, self.field_nums[0] * self.field_nums[idx]])
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

            curr_out = tf.nn.conv1d(
                dot_result, filters=self.filters[idx], stride=1, padding='VALID')
            curr_out = tf.nn.bias_add(curr_out, self.bias[idx])
            curr_out = self.activation_layers[idx](curr_out)
            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

            if self.split_half:
                if idx != len(self.layer_size) - 1:
                    next_hidden, direct_connect = tf.split(
                        curr_out, 2 * [layer_size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_result, axis=1)
        result = tf.reduce_sum(result, -1, keep_dims=False)

        return result

    def compute_output_shape(self, input_shape):
        if self.split_half:
            featuremap_num = sum(
                self.layer_size[:-1]) // 2 + self.layer_size[-1]
        else:
            featuremap_num = sum(self.layer_size)
        return None, featuremap_num

    def get_config(self, ):
        config = {'layer_size': self.layer_size, 'split_half': self.split_half, 'activation': self.activation,
                  'seed': self.seed}
        base_config = super().get_config()
        config.update(base_config)
        return config
