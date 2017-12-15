import sys

sys.path.append('./')

import time
from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf 
import cv2
import numpy as np


classes_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multi-threading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def process_predicts(resized_img, predicts):
    p_classes = predicts[0, :, :, 0:20]
    C = predicts[0, :, :, 20:22]
    coordinate = predicts[0, :, :, 22:]

    p_classes = np.reshape(p_classes, (7, 7, 1, 20))
    C = np.reshape(C, (7, 7, 2, 1))

    P = C * p_classes
    #print(P[4, 5, 1, :])

    thresh = 0.12  # threshold to select detection result
    for i in range(7):
        for j in range(7):
            temp_data = np.zeros_like(P, np.float32)
            temp_data[i, j, :, :] = P[i, j, :, :]
            position = np.argmax(temp_data)
            index = np.unravel_index(position, P.shape)
            if P[index] > thresh:
                class_num = index[3]

                coordinate = np.reshape(coordinate, (7, 7, 2, 4))

                max_coordinate = coordinate[index[0], index[1], index[2], :]

                xcenter = max_coordinate[0]
                ycenter = max_coordinate[1]
                w = max_coordinate[2]
                h = max_coordinate[3]

                xcenter = (index[1] + xcenter) * (448 / 7.0)
                ycenter = (index[0] + ycenter) * (448 / 7.0)

                w = w * 448
                h = h * 448

                # handle index out range problem
                xmin = 0 if (xcenter - w / 2.0 < 0) else (xcenter - w / 2.0)
                ymin = 0 if (ycenter - h / 2.0 < 0) else (ycenter - h / 2.0)
                xmax = resized_img.shape[0] if (xmin + w > resized_img.shape[0]) else (xmin + w)
                ymax = resized_img.shape[1] if (ymin + h > resized_img.shape[1]) else (ymin + h)

                class_name = classes_name[class_num]
                cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
                cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
    processed_image = resized_img
    return processed_image
    '''
    '''

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

if __name__ == '__main__':
    common_params = {'image_size': 448, 'num_classes': 20, 'batch_size': 1}
    net_params = {'cell_size': 7, 'boxes_per_cell': 2, 'weight_decay': 0.0005}

    net = YoloTinyNet(common_params, net_params, test=True)

    image = tf.placeholder(tf.float32, (1, 448, 448, 3))
    predicts = net.inference(image)

    sess = tf.Session()

    np_img = cv2.imread('./data/demo/001763.jpg')
    resized_img = cv2.resize(np_img, (448, 448))

    # convert to rgb image
    np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    # convert data type used in tf
    np_img = np_img.astype(np.float32)
    # data normalization and reshape to input tensor
    np_img = np_img / 255.0 * 2 - 1
    np_img = np.reshape(np_img, (1, 448, 448, 3))

    saver = tf.train.Saver(net.trainable_collection)
    saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')

    timer = Timer()
    timer.tic()
    # for i in range(1000):
    #    print(('The {:d} detection...').format(i))
    print('Procession detection...')
    np_predict = sess.run(predicts, feed_dict={image: np_img})
    timer.toc()
    print('One detection took {:.3f}s in average'.format(timer.total_time))
    processed_image = process_predicts(resized_img, np_predict)
    cv2.imwrite('redult.jpg', processed_image)
    sess.close()
