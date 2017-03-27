#!/usr/bin/python 
# -*- coding: utf-8 -*- 
import sys

sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf 
import cv2
import numpy as np

classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", 
                "bus", "car", "cat", "chair", "cow", 
                "diningtable", "dog", "horse", "motorbike", "person", 
                "pottedplant", "sheep", "sofa", "train","tvmonitor"]

common_params = { 'image_size': 448, 
                  'num_classes': 20, 
                  'batch_size':1}

net_params = {'cell_size': 7, 
              'boxes_per_cell':2, 
              'weight_decay': 0.0005}

def process_predicts(predicts):
  """
  对于规范化的输出结果对于特定的用户可能觉得不习惯，那么实现一个接口，将规范化
  的结果重新编写为用户习惯的数据类型
  """
  p_classes = predicts[0, :, :, 0:20]
  C = predicts[0, :, :, 20:22]
  coordinate = predicts[0, :, :, 22:]
  # 训练的模型设置超参数 net_params, 其中cell大小设置为7
  p_classes = np.reshape(p_classes, (7, 7, 1, 20))
  C = np.reshape(C, (7, 7, 2, 1))

  P = C * p_classes   # P size = (7, 7, 2, 20)

  #print P[5,1, 0, :]

  index = np.argmax(P)

  index = np.unravel_index(index, P.shape)

  class_num = index[3]

  coordinate = np.reshape(coordinate, (7, 7, 2, 4))

  max_coordinate = coordinate[index[0], index[1], index[2], :]

  xcenter = max_coordinate[0]
  ycenter = max_coordinate[1]
  w = max_coordinate[2]
  h = max_coordinate[3]

  xcenter = (index[1] + xcenter) * (448/7.0)
  ycenter = (index[0] + ycenter) * (448/7.0)

  w = w * 448
  h = h * 448

  xmin = xcenter - w/2.0
  ymin = ycenter - h/2.0

  xmax = xmin + w
  ymax = ymin + h

  return xmin, ymin, xmax, ymax, class_num

def main():


  net = YoloTinyNet(common_params, net_params, test=True)
  # tensorflow中声明占位符号image, 这在后面run的时候
  # feed_dict中会出现该占位符和对应的值，意思就是输入数据的来源
  image = tf.placeholder(tf.float32, (1, 448, 448, 3))
  predicts = net.inference(image)

  sess = tf.Session()

  # 转化数据格式
  np_img = cv2.imread('cat.jpg')
  resized_img = cv2.resize(np_img, (448, 448))
  np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

  np_img = np_img.astype(np.float32)
  #白化输入的数据
  np_img = np_img / 255.0 * 2 - 1
  np_img = np.reshape(np_img, (1, 448, 448, 3))

  saver = tf.train.Saver(net.trainable_collection)

  saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')
  # The optional feed_dict argument allows the caller to override 
  # the value of tensors in the graph. 
  np_predict = sess.run(predicts, feed_dict={image: np_img})

  xmin, ymin, xmax, ymax, class_num = process_predicts(np_predict)
  class_name = classes_name[class_num]
  # 绘制预测框, 输出预测类型
  cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
  cv2.putText(resized_img, 
              class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
  cv2.imwrite('cat_out.jpg', resized_img)
  sess.close()

if __name__ == '__main__':
  main()