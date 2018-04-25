from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re
import sys
import time
from datetime import datetime
from tensorflow.python import debug as tf_debug
from yolo.solver.solver import Solver


class Solver_Gan(Solver):
    """Yolo-Gan Solver
    """

    def __init__(self, dataset, d_net, g_net, common_params, solver_params):
        # process params
        self.moment = float(solver_params['moment'])
        self.learning_rate = float(solver_params['learning_rate'])
        self.batch_size = int(common_params['batch_size'])
        self.height = int(common_params['image_size'])
        self.width = int(common_params['image_size'])
        self.max_objects = int(common_params['max_objects_per_image'])
        self.pretrain_path = str(solver_params['pretrain_model_path'])
        self.train_dir = str(solver_params['train_dir'])
        self.max_iterators = int(solver_params['max_iterators'])
        #
        self.dataset = dataset
        self.g_net = g_net
        self.d_net = d_net
        # construct graph
        self.construct_graph()

    def _train(self, var_list_g):
        """Train model

        Create an optimizer and apply to all trainable variables.

        Args:
          total_loss: Total loss from net.loss()
          global_step: Integer Variable counting the number of training steps
          processed
        Returns:
          train_op: op for training
        """

        opt = tf.train.MomentumOptimizer(self.learning_rate, self.moment)
        grads = opt.compute_gradients(self.total_loss, var_list=var_list_g)

        apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

        return apply_gradient_op

    def _train_d(self, var_list_d):
        opt = tf.train.MomentumOptimizer(self.learning_rate, self.moment)
        grads = opt.compute_gradients(self.d_loss, var_list=var_list_d)
        apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)
        return apply_gradient_op

    def construct_graph(self):
        # construct graph
        with tf.variable_scope('G'):
            self.global_step = tf.Variable(0, trainable=False)
            self.images = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 3))
            self.labels = tf.placeholder(tf.float32, (self.batch_size, self.max_objects, 5))
            self.objects_num = tf.placeholder(tf.int32, (self.batch_size))

            self.predicts = self.g_net.inference(self.images)
            self.total_loss, self.nilboy = self.g_net.loss(self.predicts, self.labels, self.objects_num)
            #self.predicted_labels = self.generate_labels(self.predicts, threshhold=0.2)

            tf.summary.scalar('loss', self.total_loss)

        # Construction the net for d net
        # placeholder for d net
        with tf.variable_scope('D'):
            self.croped_image = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 3))
            self.images_original = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 3))
            # self.labels_d = tf.placeholder(tf.float32, (self.batch_size, self.max_objects, 5))
            self.d_labels = tf.placeholder(tf.float32, (self.batch_size, 1))
            self.d_predicted = self.d_net.inference(self.images_original, self.croped_image)
            # loss for  d
            self.d_loss = self.d_net.loss_d(self.d_labels, self.d_predicted)
        var_lists = tf.trainable_variables()
        self.d_var_list = [v for v in var_lists if v.name.startswith('D/')]
        self.g_var_list = [v for v in var_lists if v.name.startswith('G/')]
        self.train_op = self._train(self.g_var_list)
        self.train_op_d = self._train_d(self.d_var_list)

    def generate_labels(self, predicts, threshhold=0.2):
        # TODO: complete the generation of labels from predictions
        predicts=predicts[0]
        predicts_boxes_1 = predicts[:, :, :, 0:4]
        predicts_boxes_2 = predicts[:, :, :, 5:9]

        conf_1 = predicts[:, :, :, 4:5] * predicts[:, :, :, 10:30]
        conf_2 = predicts[:, :, :, 8:9] * predicts[:, :, :, 10:30]

        predicts_boxes_1_final = tf.concat([predicts_boxes_1, conf_1], axis=3)
        predicts_boxes_2_final = tf.concat([predicts_boxes_2, conf_2], axis=3)
        predicts_boxes = tf.stack([predicts_boxes_1_final, predicts_boxes_2_final], axis=3)
        # predicts_boxes[:, :, :, :, 4] = predicts_boxes[:, :, :, :, 4] * predicts[:, :, :, 10:30]
        return predicts_boxes

    def crop_labels(self, images, labels):
        """
        from images and labels obtain croped_images
        :param images:
        :param labels:
        :return:
        """
        # TODO:generate images from images and labels
        return images

    def solve(self):
        saver1 = tf.train.Saver(self.g_net.pretrained_collection, write_version=1)
        # saver1 = tf.train.Saver(self.net.trainable_collection)
        saver2 = tf.train.Saver(self.g_net.trainable_collection, write_version=1)

        init = tf.global_variables_initializer()

        summary_op = tf.summary.merge_all()

        sess = tf.Session()

        sess.run(init)
        # saver1.restore(sess, self.pretrain_path)

        # debug_sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
        for step in range(self.max_iterators):
            start_time = time.time()
            np_images, np_labels, np_objects_num = self.dataset.batch()

            _, loss_value, nilboy = sess.run([self.train_op, self.total_loss, self.nilboy],
                                             feed_dict={self.images: np_images, self.labels: np_labels,
                                                        self.objects_num: np_objects_num})
            # loss_value, nilboy = sess.run([self.total_loss, self.nilboy], feed_dict={self.images: np_images, self.labels: np_labels, self.objects_num: np_objects_num})
            # predicted = sess.run(self.predicts, feed_dict={self.images: np_images})
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = self.dataset.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, g_loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

                sys.stdout.flush()
            if step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict={self.images: np_images, self.labels: np_labels,
                                                              self.objects_num: np_objects_num})
                summary_writer.add_summary(summary_str, step)
            if step % 5000 == 0:
                saver2.save(sess, self.train_dir + '/model.ckpt', global_step=step)

            # train the d net
            start_time = time.time()
            croped_images = self.crop_labels(np_images, np_labels)
            labels_positive = np.ones([16, 1])

            _, d_loss_positive = sess.run([self.train_op_d, self.d_loss],
                                          feed_dict={self.images_original: np_images, self.croped_image: croped_images,
                                                     self.d_labels: labels_positive})
            assert not np.isnan(d_loss_positive), 'Model diverged with loss = NaN'
            # train d net with negtive samples

            labels_negtive = np.zeros([16, 1])
            predicted_g = sess.run([self.predicts], feed_dict={self.images: np_images})
            predicted_labels = self.generate_labels(predicted_g)
            croped_images_g = self.crop_labels(np_images, predicted_labels)
            _, d_loss_negtive = sess.run([self.train_op_d, self.d_loss], feed_dict={self.images_original: np_images
                , self.croped_image: croped_images_g,
                                                                                    self.d_labels: labels_negtive})
            assert not np.isnan(d_loss_negtive), 'Model diverged with loss = NaN'
            end_time = time.time()

            duration = end_time - start_time
            if step % 10 == 0:
                num_examples_per_step = self.dataset.batch_size * 2
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, d_loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, (d_loss_negtive + d_loss_positive) / 2,
                                    examples_per_sec, sec_per_batch))

        sess.close()
