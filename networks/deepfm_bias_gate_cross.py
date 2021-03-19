from typing import List, Dict
import tensorflow as tf

from layers import fm_net, DeepLayer
from networks.base_network import Network
from layers.core import SafeKEmbedding, CategorizeLayer, CategorizeEmbeddingLayer, sparse_embedding
from layers.sequence import SequencePoolingLayer, SequenceWeightedPoolingLayer, DinSequencePoolingLayer

class DeepFM_bias_gate_cross(Network):
    def _build(self):
        self.deep = DeepLayer(
            [int(i) for i in self._hidden.split(',')],
            activation=self._flags.activation,
            l2_reg=self._flags.l2,
            dropout_rate=self._flags.dropout,
            use_bn=self._flags.use_bn
        )
        self.bias_net = DeepLayer(
            [128,128,8],
            activation=self._flags.activation,
            l2_reg=self._flags.l2,
            dropout_rate=self._flags.dropout,
            use_bn=self._flags.use_bn
        )
        super()._build()
    def call(self, dense, embeddings, is_training):
        print("call") 
       
    def _do(self, dense, embeddings, linear_weight, is_training):
        with tf.name_scope(name=self._name):
            deep_in = tf.concat(dense + [tf.squeeze(emb, [1]) for emb in embeddings], -1)
            dense_out = self.deep(deep_in)
            fm_in = tf.concat(embeddings, 1)
            fm_out = fm_net(fm_in)
            bias_weight = tf.concat(linear_weight, -1)
            bias_out = self.bias_net(bias_weight)
            bias_bias = tf.expand_dims(tf.reduce_sum(bias_weight,axis=1),axis=-1)
            return tf.concat((dense_out, fm_out, bias_out, bias_bias) , -1)

    def _build_gate(self, features_embeding, gate_num):
        features_list = [x for x in self._flags.gate_input_feature_list.split(",")]
        f_emb = tf.concat([tf.squeeze(features_embeding[key], [1]) for key in features_list], -1) 
        inputs = tf.keras.layers.Dense(100, activation=tf.nn.relu)(f_emb)
        gate = tf.keras.layers.Dense(gate_num, activation=tf.nn.sigmoid)(inputs)
        gate = tf.expand_dims(gate, axis=1)
        return gate

    def build_features(self, features):
        if self._flags.feature_influence and self._flags.feature_influence in features:
            features[self._flags.feature_influence] = tf.cast(features[self._flags.feature_influence] * 0,
                                                              features[self._flags.feature_influence].dtype)
        emb_dim = {k: self._calculus_emb_dim(v) for k, v in self.voc_size.items()}
        emb_layer = SafeKEmbedding if self._flags.safe_embedding else tf.keras.layers.Embedding
        emb_map = {name: emb_layer(self.voc_size[name], emb_dim[name], mask_zero=True, name='embedding_{}'.format(name))
                   for name in emb_dim.keys() }
        weight_bias_map = {name: emb_layer(self.voc_size[name], 1, mask_zero=True, name='linear_weight_{}'.format(name))
                   for name in emb_dim.keys() }
        #emb_map["user_id"] = SafeKEmbedding(self.voc_size["user_id"], emb_dim["user_id"], mask_zero=True, name='embedding_{}'.format("user_id"),embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001, seed=None)) 
         
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
        pooled_embeddings = self._build_sequence(sequence_embeddings,
                                                 sequence_masks,
                                                 sequence_weight,
                                                 categorical_embeddings, attention_map, self._flags.combiner)
        
        categorical_weight_bias = {name: weight_bias_map[self._dataset.emb_name[name]](features[name])
                                  for name in self.categorical_list if name in features}
        sequence_weight_bias = {name: weight_bias_map[self._dataset.emb_name[name]](features[name])
                               for name in self._dataset.sequence_list if name in features}
        
        linear_weight_bias = [ tf.squeeze(categorical_weight_bias[key], axis=-1) for key in sorted(categorical_weight_bias.keys())]
        linear_weight_bias.extend([tf.squeeze(sequence_weight_bias[key], axis=-1) for key in sorted(sequence_weight_bias.keys())])
        


        numerical = self._build_dense([features[name] for name in sorted(self._dataset.numerical_list)
                                       if name in features])
        vectors = [features[name] for name in sorted(self._dataset.vector_list) if name in features]

        dense = numerical + vectors
        #user_features_list = ["user_id","user_level","net","device_model", "real_play_list_size","real_like_list_size", "real_share_list_size", "real_download_list_size","real_follow_list_size", "real_follow_author_list_size", "real_finish_list_size", "manufacturer", "release_channel", "dpi", "screen_width", "screen_height", "real_play_list","real_finish_list","user_real_labels_pos","user_real_labels_neg", "req_province", "user_real_labels2_pos","user_real_labels2_neg", "phone_brand_price", "phone_brand", "is_new_user", "user_short_labels", "request_cnt"]
       
        ###user_cross_feature = [x for x in self._flags.user_cross_feature.split(",")] 
        ###item_cross_features = [x for x in self._flags.item_cross_feature.split(",")]
        ###user_features_list_gate_controled = [x for x in self._flags.user_features_list_gate_controled.split(",")]
        #user_features_list = ["user_id","user_level", "real_play_list_size","real_like_list_size", "real_share_list_size", "real_download_list_size","real_follow_list_size", "real_follow_author_list_size", "real_finish_list_size", "real_play_list","real_finish_list","user_real_labels_pos","user_real_labels_neg", "req_province", "user_real_labels2_pos","user_real_labels2_neg",  "is_new_user", "user_short_labels"]
        #item_features_list = ["tags","duration", "source_user", "final_label1","final_label2", "effective_rate","finish_rate", "con_effective_rate", "con_finish_rate","hashtags","bgm_id","author_id","publish_dispersed_day","countries","langs"]

        #user_features_list_gate = ["user_id", "real_play_list","real_finish_list","user_real_labels_pos","user_real_labels_neg", "user_real_labels2_pos","user_real_labels2_neg", "user_short_labels"]       
 
        sum_pooled_embeddings = self._build_sequence(sequence_embeddings,
                                                 sequence_masks,
                                                 sequence_weight,
                                                 categorical_embeddings,
                                                 attention_map, "sum")
       
        ###embeding_user = [ categorical_embeddings[key] for  key in user_cross_feature  if key in categorical_embeddings.keys() ]
        ###tf.logging.warn(embeding_user)
        ###tf.logging.warn(sum_pooled_embeddings.keys())
        ###embeding_user.extend([ sum_pooled_embeddings[key] for  key in user_cross_feature  if key in sum_pooled_embeddings.keys() ])
        ###tf.logging.warn(embeding_user)
        ###embeding_item = [ categorical_embeddings[key] for  key in item_cross_features  if key in categorical_embeddings.keys() ]
        ###embeding_item.extend([ sum_pooled_embeddings[key] for  key in item_cross_features  if key in sum_pooled_embeddings.keys() ])
        ###sum_emb_user = tf.reduce_sum(tf.transpose(embeding_user, perm=(1,0,2,3)), axis=1)
        ###tf.logging.warn("sum_emb_user")
        ###tf.logging.warn(sum_emb_user)
        ###sum_emb_item = tf.reduce_sum(tf.transpose(embeding_item, perm=(1,0,2,3)), axis=1)
        ###tf.logging.warn("sum_emb_item")
        ###tf.logging.warn(sum_emb_item)
        ###cross_emb = sum_emb_user * sum_emb_item
        ###tf.logging.warn("cross_emb")
        ###tf.logging.warn(cross_emb)
        ### 
        ###user_id_feature_gate = self._build_gate(categorical_embeddings,len(user_features_list_gate_controled)) 
        ####categorical_embeddings["user_id"] = user_id_feature_gate * categorical_embeddings["user_id"]
        ###i = 0
        ###for key in user_features_list_gate_controled:
        ###    if key in categorical_embeddings.keys():
        ###       tf.logging.warn("slicing")
        ###       tf.logging.warn(tf.slice(user_id_feature_gate,[0,0,i], [-1,1,1]))
        ###       categorical_embeddings[key] = tf.slice(user_id_feature_gate,[0,0,i], [-1,1,1]) * categorical_embeddings[key]
        ###    if key in pooled_embeddings.keys():
        ###       tf.logging.warn("slicing")
        ###       tf.logging.warn(tf.slice(user_id_feature_gate,[0,0,i], [-1,1,1]))
        ###       pooled_embeddings[key] = tf.slice(user_id_feature_gate,[0,0,i], [-1,1,1]) * pooled_embeddings[key]
        ###    i += 1


        embeddings = [categorical_embeddings[f] for f in sorted(categorical_embeddings.keys())]
        embeddings.extend([pooled_embeddings[f] for f in sorted(pooled_embeddings.keys())])
        embeddings.extend([numerical_embeddings[f] for f in sorted(numerical_embeddings.keys())])
        embeddings.extend([varlen_embeddings[f] for f in sorted(varlen_embeddings.keys())])
        tf.logging.warn("embeddings")
        tf.logging.warn(embeddings)
        ###embeddings.append(cross_emb)
        return dense, embeddings, linear_weight_bias

    def _build_sequence(self, sequence_embeddings: Dict[str, tf.Tensor],
                        sequence_masks: Dict[str, tf.Tensor],
                        weights: Dict[str, tf.Tensor],
                        categorical_embeddings: Dict[str, tf.Tensor],
                        attention_map: Dict[str, str], mode :str):
        pooled_embeddings = dict()
        pooling_layer = SequencePoolingLayer(mode=mode, supports_masking=True)
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
