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
import tflearn
import tensorflow as tf
from layers import GraphProjection, GraphPooling, AttentionGraphConvolution, GraphRebuild
from losses import mesh_loss, laplace_loss

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('k_graph', 8, 'Number of neighbors for dynamic graph')

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.output1 = None
        self.output2 = None
        self.output3 = None
        self.output1_2 = None
        self.output2_2 = None

        self.loss = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        print("Building model %s..." % self.name)
        # print("len(self.activations)", len(self.activations))
        # print("(self.activations.shape)", [a.get_shape() for a in self.activations])
        #with tf.device('/gpu:0'):
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential resnet model
        eltwise = [3,5,7,9,11,13, 19,21,23,25,27,29, 35,37,39,41,43,45]
        concat = [15, 31]
        self.activations.append(self.inputs)
        for idx,layer in enumerate(self.layers):
            hidden = layer(self.activations[-1])
            # skip residual/concat removed due to dimension mismatch between image features and graph features
            self.activations.append(hidden)

        # print("after build")
        # print("len(self.activations)", len(self.activations))
        # print("(self.activations.shape)", [a.get_shape() for a in self.activations])
        # #with tf.device('/gpu:0'):
        # fixed coordinate layer indices: block1 @16, block2 @32, block3 @-1
        self.output1 = self.activations[16]
        unpool_layer = GraphPooling(placeholders=self.placeholders, pool_id=1)
        self.output1_2 = unpool_layer(self.output1)
        self.output2 = self.activations[32]
        unpool_layer = GraphPooling(placeholders=self.placeholders, pool_id=2)
        self.output2_2 = unpool_layer(self.output2)
        self.output3 = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
        # Build loss and training operation
        self._loss()
        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "Data/checkpoint/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "Data/checkpoint/%s.ckpt" % self.name
        #save_path = "checks/tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

class GCN(Model):
    def __init__(self, placeholders, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _build(self):
        self.build_cnn18() #update image feature
        # first dynamic graph reconstruction
        self.layers.append(GraphRebuild(placeholders=self.placeholders, gcn_block_id=1, k=FLAGS.k_graph))
        # first project block
        self.layers.append(GraphProjection(placeholders=self.placeholders))
        self.layers.append(AttentionGraphConvolution(input_dim=FLAGS.feat_dim,
                                                    output_dim=FLAGS.hidden,
                                                    gcn_block_id=1,
                                                    placeholders=self.placeholders, logging=self.logging))
        for _ in range(12):
            self.layers.append(AttentionGraphConvolution(input_dim=FLAGS.hidden,
                                                        output_dim=FLAGS.hidden,
                                                        gcn_block_id=1,
                                                        placeholders=self.placeholders, logging=self.logging))
        self.layers.append(AttentionGraphConvolution(input_dim=FLAGS.hidden,
                                                    output_dim=FLAGS.coord_dim,
                                                    act=lambda x: x,
                                                    gcn_block_id=1,
                                                    placeholders=self.placeholders, logging=self.logging))
        # before second block dynamic graph
        self.layers.append(GraphRebuild(placeholders=self.placeholders, gcn_block_id=2, k=FLAGS.k_graph))
        self.layers.append(GraphProjection(placeholders=self.placeholders))
        # block2 first AGC should consume projected image features of dimension feat_dim
        self.layers.append(AttentionGraphConvolution(input_dim=FLAGS.feat_dim,
                                                    output_dim=FLAGS.hidden,
                                                    gcn_block_id=2,
                                                    placeholders=self.placeholders, logging=self.logging))
        for _ in range(12):
            self.layers.append(AttentionGraphConvolution(input_dim=FLAGS.hidden,
                                                        output_dim=FLAGS.hidden,
                                                        gcn_block_id=2,
                                                        placeholders=self.placeholders, logging=self.logging))
        self.layers.append(AttentionGraphConvolution(input_dim=FLAGS.hidden,
                                                    output_dim=FLAGS.coord_dim,
                                                    act=lambda x: x,
                                                    gcn_block_id=2,
                                                    placeholders=self.placeholders, logging=self.logging))
        # before third block dynamic graph
        self.layers.append(GraphRebuild(placeholders=self.placeholders, gcn_block_id=3, k=FLAGS.k_graph))
        self.layers.append(GraphProjection(placeholders=self.placeholders))
        # block3 first AGC should consume projected image features of dimension feat_dim
        self.layers.append(AttentionGraphConvolution(input_dim=FLAGS.feat_dim,
                                                    output_dim=FLAGS.hidden,
                                                    gcn_block_id=3,
                                                    placeholders=self.placeholders, logging=self.logging))
        for _ in range(12):
            self.layers.append(AttentionGraphConvolution(input_dim=FLAGS.hidden,
                                                        output_dim=FLAGS.hidden,
                                                        gcn_block_id=3,
                                                        placeholders=self.placeholders, logging=self.logging))
        self.layers.append(AttentionGraphConvolution(input_dim=FLAGS.hidden,
                                                    output_dim=int(FLAGS.hidden/2),
                                                    gcn_block_id=3,
                                                    placeholders=self.placeholders, logging=self.logging))
        self.layers.append(AttentionGraphConvolution(input_dim=int(FLAGS.hidden/2),
                                                    output_dim=FLAGS.coord_dim,
                                                    act=lambda x: x,
                                                    gcn_block_id=3,
                                                    placeholders=self.placeholders, logging=self.logging))

    def build_cnn18(self):
        x=self.placeholders['img_inp']
        x=tf.expand_dims(x, 0)
#224 224
        x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x0=x
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#112 112
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x1=x
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#56 56
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x2=x
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#28 28
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x3=x
        x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#14 14
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x4=x
        x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#7 7
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x5=x
#updata image feature
        self.placeholders.update({'img_feat': [tf.squeeze(x2), tf.squeeze(x3), tf.squeeze(x4), tf.squeeze(x5)]})

    def _loss(self):
        # mesh reconstruction and Laplacian losses (matching p2m/models)
        self.loss += mesh_loss(self.output1, self.placeholders, 1)
        self.loss += mesh_loss(self.output2, self.placeholders, 2)
        self.loss += mesh_loss(self.output3, self.placeholders, 3)
        # Laplacian regularization temporarily disabled (causing shape mismatch)
        # self.loss += .1 * laplace_loss(self.inputs, self.output1, self.placeholders, 1)
        # self.loss += laplace_loss(self.output1_2, self.output2, self.placeholders, 2)
        # self.loss += laplace_loss(self.output2_2, self.output3, self.placeholders, 3)
