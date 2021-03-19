from typing import List, Dict

import tensorflow as tf

from layers.core import SafeKEmbedding, CategorizeLayer, CategorizeEmbeddingLayer, sparse_embedding
from layers.sequence import SequencePoolingLayer, SequenceWeightedPoolingLayer, DinSequencePoolingLayer


class Network(object):
    def __init__(self, flags, dataset, name, **kwargs):
        self._flags = flags
        self._hidden = kwargs['hidden'] if 'hidden' in kwargs else flags.hidden
        self._dataset = dataset
        self._name = name
        self.numerical_list = self._dataset.numerical_list[:]
        self.voc_size = self._dataset.voc_size.copy()
        self.categorical_list = self._dataset.categorical_list[:]
        self.is_build = False

    def __call__(self, dense, embeddings, is_training):
        if not self.is_build:
            self._build()
        return self.call(dense, embeddings, is_training)

    def _build(self):
        self.is_build = True

    def call(self, dense, embeddings, is_training):
        raise NotImplementedError()

    def build_features(self, features):
        if self._flags.feature_influence and self._flags.feature_influence in features:
            features[self._flags.feature_influence] = tf.cast(features[self._flags.feature_influence] * 0,
                                                              features[self._flags.feature_influence].dtype)
        emb_dim = {k: self._calculus_emb_dim(v) for k, v in self.voc_size.items()}
        # emb_dim = {k: 8 for k, v in self.voc_size.items()}
        emb_layer = SafeKEmbedding if self._flags.safe_embedding else tf.keras.layers.Embedding
        emb_map = {name: emb_layer(self.voc_size[name], emb_dim[name], mask_zero=True, name='embedding_{}'.format(name))
                   for name in emb_dim.keys() if name in features}

        attention_map = dict()
        for item in self._flags.attention_cols.split(','):
            try:
                key, query = item.split(':')
            except Exception:
                continue
            attention_map[key] = query

        numerical_embeddings = dict()
        if self._flags.categorize_numerical == 'bucket':
            features = self._categorize_numerical(features)
            numerical_embeddings = {name: emb_map[self._dataset.emb_name[name]](features[name])
                                    for name in self.numerical_list if name in features}
            self.numerical_list = list()
        elif self._flags.categorize_numerical == 'embedding':
            numerical_embeddings = {name: CategorizeEmbeddingLayer(emb_dim[name], name=name)(features[name])
                                    for name in self.numerical_list if name in features}
            self.numerical_list = list()
        categorical_embeddings = {name: emb_map[self._dataset.emb_name[name]](features[name])
                                  for name in self.categorical_list if name in features}
        varlen_embeddings = {name: sparse_embedding(features[name], name, self.voc_size[name], emb_dim[name])
                             for name in self._dataset.varlen_list}
        sequence_embeddings = {name: emb_map[self._dataset.emb_name[name]](features[name])
                               for name in self._dataset.sequence_list if name in features}
        sequence_masks = {name: emb_map[self._dataset.emb_name[name]].compute_mask(features[name])
                          for name in self._dataset.sequence_list if name in features}
        sequence_weight = {name: features[name] for name in self._dataset.weight_sequence_list if name in features}
        
        with tf.variable_scope("pooling",reuse=True):
            pooled_embeddings = self._build_sequence(sequence_embeddings,
                                                 sequence_masks,
                                                 sequence_weight,
                                                 categorical_embeddings, attention_map)

        numerical = self._build_dense([features[name] for name in sorted(self._dataset.numerical_list)
                                       if name in features])
        vectors = [features[name] for name in sorted(self._dataset.vector_list) if name in features]

        dense = numerical + vectors
        embeddings = [categorical_embeddings[f] for f in sorted(categorical_embeddings.keys())]
        embeddings.extend([pooled_embeddings[f] for f in sorted(pooled_embeddings.keys())])
        embeddings.extend([numerical_embeddings[f] for f in sorted(numerical_embeddings.keys())])
        embeddings.extend([varlen_embeddings[f] for f in sorted(varlen_embeddings.keys())])
        return dense, embeddings

    @staticmethod
    def _build_dense(numerical: List[tf.Tensor]):
        if not numerical:
            return list()
        numerical = tf.concat(numerical, -1) if len(numerical) > 1 else numerical[0]
        return [tf.keras.layers.BatchNormalization()(numerical)]

    def _build_sequence(self, sequence_embeddings: Dict[str, tf.Tensor],
                        sequence_masks: Dict[str, tf.Tensor],
                        weights: Dict[str, tf.Tensor],
                        categorical_embeddings: Dict[str, tf.Tensor],
                        attention_map: Dict[str, str]):
        pooled_embeddings = dict()
        pooling_layer = SequencePoolingLayer(mode=self._flags.combiner, supports_masking=True)
        weighted_pooling_layer = SequenceWeightedPoolingLayer()
        din_polling_layers = {key: DinSequencePoolingLayer(weight_norm=self._flags.norm_attention, name=key)
                              for key in attention_map.keys()}
        for key, embedding in sequence_embeddings.items():
            mask = sequence_masks[key]
            if key + '_weight' in weights:
                seq_emb = weighted_pooling_layer(embedding, weights=weights[key + '_weight'])
            elif key in attention_map:
                seq_emb = din_polling_layers[key]([categorical_embeddings[attention_map[key]], embedding], mask=mask)
            else:
                seq_emb = pooling_layer(embedding, mask=mask)
            pooled_embeddings[key] = seq_emb
        return pooled_embeddings

    def _calculus_emb_dim(self, voc_size):
        emb_dim = self._flags.fixed_emb_dim
        return emb_dim if emb_dim else int(voc_size ** 0.25 * self._flags.emb_factor)

    def _categorize_numerical(self, features):
        categorized_features = dict()
        self.voc_size.update(self._dataset.numerical_voc)
        for name, tensor in features.items():
            if name in self.numerical_list:
                tensor = CategorizeLayer(self.voc_size[name])(tensor)
            categorized_features[name] = tensor
        return categorized_features
