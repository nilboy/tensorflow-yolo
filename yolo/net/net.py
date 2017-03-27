#!/usr/bin/python 
# -*- coding: utf-8 -*- 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

class Net(object):
  """Base Net class 
  """
  def __init__(self, common_params, net_params):
    """
    common_params: a params dict
    net_params: a params dict
    """
    #pretrained variable collection
    self.pretrained_collection = []
    #trainable variable collection
    self.trainable_collection = []

  def _variable_on_cpu(self, name, shape, initializer, pretrain=True, train=True):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the Variable
      shape: list of ints
      initializer: initializer of Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
      if pretrain:
        self.pretrained_collection.append(var)
      if train:
        self.trainable_collection.append(var)
    return var 


  def _variable_with_weight_decay(self, name, shape, stddev, wd, pretrain=True, train=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with truncated normal distribution
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable 
      shape: list of ints
      stddev: standard devision of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight 
      decay is not added for this Variable.
   Returns:
      Variable Tensor 
    """
    var = self._variable_on_cpu(name, shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32), pretrain, train)
    if wd is not None:
      weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)
    return var 

  def conv2d(self, scope, input, kernel_size, stride=1, pretrain=True, train=True):
    """convolutional layer

    Args:
      input: 4-D tensor [batch_size, height, width, depth]
      scope: variable_scope name 
      kernel_size: [k_height, k_width, in_channel, out_channel]
      stride: int32
    Return:
      output: 4-D tensor [batch_size, height/stride, width/stride, out_channels]
    """
    with tf.variable_scope(scope) as scope:
      # 初始化权重的kernel
      kernel = self._variable_with_weight_decay('weights', 
                                      shape=kernel_size,
                                      stddev=5e-2,
                                      wd=self.weight_decay, pretrain=pretrain, train=train)
      conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding='SAME')

      # biases 初始化采用常数 0.0 初始化
      biases = self._variable_on_cpu('biases', kernel_size[3:], tf.constant_initializer(0.0), pretrain, train)
      conv1 = tf.nn.bias_add(conv, biases)
      output = self.leaky_relu(conv1)

    return output


  def max_pool(self, input, kernel_size, stride):
    """max_pool layer

    Args:
      input: 4-D tensor [batch_zie, height, width, depth]
      kernel_size: [k_height, k_width]
      stride: int32
    Return:
      output: 4-D tensor [batch_size, height/stride, width/stride, depth]
    """
    return tf.nn.max_pool(input, ksize=[1, kernel_size[0], kernel_size[1], 1], strides=[1, stride, stride, 1],
                  padding='SAME')

  def local(self, scope, _input, in_dimension, out_dimension, leaky=True, pretrain=True, train=True):
    """Fully connection layer

    Args:
      scope: variable_scope name
      input: [batch_size, ???]
      out_dimension: int32
    Return:
      output: 2-D tensor [batch_size, out_dimension]
    """
    with tf.variable_scope(scope) as scope:
      reshape = tf.reshape(_input, [tf.shape(_input)[0], -1])

      weights = self._variable_with_weight_decay('weights', shape=[in_dimension, out_dimension],
                          stddev=0.04, wd=self.weight_decay, pretrain=pretrain, train=train)
      biases = self._variable_on_cpu('biases', [out_dimension], tf.constant_initializer(0.0), pretrain, train)
      local = tf.matmul(reshape, weights) + biases

      if leaky:
        local = self.leaky_relu(local)
      else:
        local = tf.identity(local, name=scope.name)

    return local

  def leaky_relu(self, x, alpha=0.1, dtype=tf.float32):
    """leaky relu 
    if x > 0:
      return x
    else:
      return alpha * x
    Args:
      x : Tensor
      alpha: float
    Return:
      y : Tensor
    """
    x = tf.cast(x, dtype=dtype)
    # 对输入的特征向量进行leaky_relu
    # 其中对>0的数据采用直接激活的方式，对小于0的数据采用leaky激活方式
    # 此处实现值得学习和借鉴
    bool_mask = (x > 0)
    mask = tf.cast(bool_mask, dtype=dtype)
    return 1.0 * mask * x + alpha * (1 - mask) * x

  def inference(self, images):
    """Build the yolo model

    Args:
      images:  4-D tensor [batch_size, image_height, image_width, channels]
    Returns:
      predicts: 4-D tensor [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell]
    """
    raise NotImplementedError

  def loss(self, predicts, labels, objects_num):
    """Add Loss to all the trainable variables

    Args:
      predicts: 4-D tensor [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
      ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
      labels  : 3-D tensor of [batch_size, max_objects, 5]
      objects_num: 1-D tensor [batch_size]
    """
    raise NotImplementedError

'''
## weight decay：
在机器学习或者模式识别中，会出现overfitting，而当网络逐渐overfitting时网络
权值逐渐变大，因此，为了避免出现overfitting,会给误差函数添加一个惩罚项，常用
的惩罚项是所有权重的平方乘以一个衰减常量之和。其用来惩罚大的权值。
权值衰减惩罚项使得权值收敛到较小的绝对值，而惩罚大的权值。因为大的权值会使得
系统出现过拟合，降低其泛化性能。
'''