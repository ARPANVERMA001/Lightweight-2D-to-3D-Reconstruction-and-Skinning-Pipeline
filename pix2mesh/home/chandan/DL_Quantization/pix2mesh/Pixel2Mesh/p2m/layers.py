# -*- coding: utf-8 -*-
#  Copyright (C) 2019 Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, Yu-Gang Jiang, Fudan University
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import division
from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def project(img_feat, x, y, dim):
	x1 = tf.floor(x)
	x2 = tf.minimum(tf.ceil(x), tf.cast(tf.shape(img_feat)[0], tf.float32) - 1)
	y1 = tf.floor(y)
	y2 = tf.minimum(tf.ceil(y), tf.cast(tf.shape(img_feat)[1], tf.float32) - 1)
	Q11 = tf.gather_nd(img_feat, tf.stack([tf.cast(x1,tf.int32), tf.cast(y1,tf.int32)],1))
	Q12 = tf.gather_nd(img_feat, tf.stack([tf.cast(x1,tf.int32), tf.cast(y2,tf.int32)],1))
	Q21 = tf.gather_nd(img_feat, tf.stack([tf.cast(x2,tf.int32), tf.cast(y1,tf.int32)],1))
	Q22 = tf.gather_nd(img_feat, tf.stack([tf.cast(x2,tf.int32), tf.cast(y2,tf.int32)],1))

	weights = tf.multiply(tf.subtract(x2,x), tf.subtract(y2,y))
	Q11 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q11)

	weights = tf.multiply(tf.subtract(x,x1), tf.subtract(y2,y))
	Q21 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q21)

	weights = tf.multiply(tf.subtract(x2,x), tf.subtract(y,y1))
	Q12 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q12)

	weights = tf.multiply(tf.subtract(x,x1), tf.subtract(y,y1))
	Q22 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q22)

	outputs = tf.add_n([Q11, Q21, Q12, Q22])
	return outputs

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=False,
                 sparse_inputs=False, act=tf.nn.relu, bias=True, gcn_block_id=1,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        if gcn_block_id == 1:
			self.support = placeholders['support1']
        elif gcn_block_id == 2:
			self.support = placeholders['support2']
        elif gcn_block_id == 3:
			self.support = placeholders['support3']
			
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = 3#placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class GraphPooling(Layer):
	"""Graph Pooling layer."""
	def __init__(self, placeholders, pool_id=1, **kwargs):
		super(GraphPooling, self).__init__(**kwargs)

		self.pool_idx = placeholders['pool_idx'][pool_id-1]

	def _call(self, inputs):
		X = inputs

		add_feat = (1/2.0) * tf.reduce_sum(tf.gather(X, self.pool_idx), 1)
		outputs = tf.concat([X, add_feat], 0)

		return outputs

class GraphProjection(Layer):
	"""Graph Pooling layer."""
	def __init__(self, placeholders, **kwargs):
		super(GraphProjection, self).__init__(**kwargs)

		self.img_feat = placeholders['img_feat']

	'''
	def _call(self, inputs):
		coord = inputs
		X = inputs[:, 0]
		Y = inputs[:, 1]
		Z = inputs[:, 2]

		#h = (-Y)/(-Z)*248 + 224/2.0 - 1
		#w = X/(-Z)*248 + 224/2.0 - 1 [28,14,7,4]
		h = 248.0 * tf.divide(-Y, -Z) + 112.0
		w = 248.0 * tf.divide(X, -Z) + 112.0

		h = tf.minimum(tf.maximum(h, 0), 223)
		w = tf.minimum(tf.maximum(w, 0), 223)
		indeces = tf.stack([h,w], 1)

		idx = tf.cast(indeces/(224.0/56.0), tf.int32)
		out1 = tf.gather_nd(self.img_feat[0], idx)
		idx = tf.cast(indeces/(224.0/28.0), tf.int32)
		out2 = tf.gather_nd(self.img_feat[1], idx)
		idx = tf.cast(indeces/(224.0/14.0), tf.int32)
		out3 = tf.gather_nd(self.img_feat[2], idx)
		idx = tf.cast(indeces/(224.0/7.00), tf.int32)
		out4 = tf.gather_nd(self.img_feat[3], idx)

		outputs = tf.concat([coord,out1,out2,out3,out4], 1)
		return outputs
	'''
	def _call(self, inputs):
		coord = inputs
		X = inputs[:, 0]
		Y = inputs[:, 1]
		Z = inputs[:, 2]

		h = 250 * tf.divide(-Y, -Z) + 112
		w = 250 * tf.divide(X, -Z) + 112

		h = tf.minimum(tf.maximum(h, 0), 223)
		w = tf.minimum(tf.maximum(w, 0), 223)

		x = h/(224.0/56)
		y = w/(224.0/56)
		out1 = project(self.img_feat[0], x, y, 64)

		x = h/(224.0/28)
		y = w/(224.0/28)
		out2 = project(self.img_feat[1], x, y, 128)

		x = h/(224.0/14)
		y = w/(224.0/14)
		out3 = project(self.img_feat[2], x, y, 256)

		x = h/(224.0/7)
		y = w/(224.0/7)
		out4 = project(self.img_feat[3], x, y, 512)
		outputs = tf.concat([coord,out1,out2,out3,out4], 1)
		return outputs

class AttentionGraphConvolution(Layer):
    """Attention-augmented graph convolution layer stub."""
    def __init__(self, input_dim, output_dim, placeholders,
                 gcn_block_id=1, k=8, attn_dim=None, act=tf.nn.relu, **kwargs):
        super(AttentionGraphConvolution, self).__init__(**kwargs)
        self.act = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.block_id = gcn_block_id
        self.placeholders = placeholders
        self.k = k
        self.attn_dim = attn_dim or input_dim // 2
        with tf.variable_scope(self.name + '_vars'):
            self.vars['Wq'] = glorot([input_dim, self.attn_dim], name='Wq')
            self.vars['Wk'] = glorot([input_dim, self.attn_dim], name='Wk')
            self.vars['Wv'] = glorot([input_dim, output_dim], name='Wv')
    def _call(self, inputs):
        # inputs: [N, input_dim]
        x = inputs
        # dynamic number of nodes
        N = tf.shape(x)[0]
        # linear projections
        Q = dot(x, self.vars['Wq'])  # [N, attn_dim]
        K = dot(x, self.vars['Wk'])  # [N, attn_dim]
        V = dot(x, self.vars['Wv'])  # [N, output_dim]
        edges = self.placeholders['edges'][self.block_id-1]  # shape [E,2]
        idx_i = edges[:,0]
        idx_j = edges[:,1]
        Qi = tf.gather(Q, idx_i)
        Kj = tf.gather(K, idx_j)
        # use gather_nd for gradient-safe row lookup
        idx_j_exp = tf.stack([idx_j, tf.zeros_like(idx_j)], axis=1)
        Vj = tf.gather_nd(V, idx_j_exp)
        # attention logits
        logits = tf.reduce_sum(Qi * Kj, axis=1) / tf.sqrt(tf.cast(self.attn_dim, tf.float32))
        # segment-softmax per source idx_i
        exp_logits = tf.exp(logits)
        # sum of exp per source node
        seg_sum = tf.math.unsorted_segment_sum(exp_logits, idx_i, N)
        # normalize
        alpha = exp_logits / tf.gather(seg_sum, idx_i)
        # weighted neighbor value sum (simple scatter-add)
        weighted = Vj * tf.expand_dims(alpha, -1)
        # sum contributions per vertex (naive: scatter_add)
        output = tf.math.unsorted_segment_sum(weighted, idx_i, N)
        return self.act(output)

class GraphRebuild(Layer):
    """Dynamic k-NN graph reconstruction layer."""
    def __init__(self, placeholders, gcn_block_id, k=8, **kwargs):
        super(GraphRebuild, self).__init__(**kwargs)
        self.placeholders = placeholders
        self.block_id = gcn_block_id
        self.k = k
    def _call(self, inputs):
        # inputs: [N,3] vertex coordinates
        coords = inputs
        N = tf.shape(coords)[0]
        # pairwise squared distances
        coords_exp1 = tf.expand_dims(coords, 1)        # [N,1,3]
        coords_exp2 = tf.expand_dims(coords, 0)        # [1,N,3]
        dists = tf.reduce_sum(tf.square(coords_exp1 - coords_exp2), axis=2)  # [N,N]
        # find k+1 smallest (including self)
        neg_dists = -dists
        vals, idx = tf.nn.top_k(neg_dists, k=self.k+1)  # [N,k+1]
        # drop self index at pos 0
        neigh_idx = idx[:,1:]  # [N,k]
        # build edge list of shape [N*k,2]
        # repeat each index self.k times: use tile and reshape instead of tf.repeat
        src = tf.reshape(tf.tile(tf.expand_dims(tf.range(N), 1), [1, self.k]), [-1])
        dst = tf.reshape(neigh_idx, [-1])
        edges = tf.stack([src, dst], axis=1)
        # restore mutation of placeholders for dynamic edges (original behavior)
        self.placeholders['edges'][self.block_id-1] = edges
        return inputs

