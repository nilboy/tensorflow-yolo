import cv2
import numpy as np
import tensorflow as tf
from yolo.net.yolo_tiny_net import YoloTinyNet

CLASS_NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

class PredictionWindow(object):
	""" opens window showing the models predictions """
	def __init__(self, common_params, net_params):
		self.net = YoloTinyNet(common_params, net_params, test=True)

		self.image_size = common_params['image_size']
		self.image = tf.placeholder(tf.float32, (1, self.image_size, self.image_size, 3))
		self.predicts = self.net.inference(self.image)

	def run(self, source_callback):
		sess = tf.Session()

		saver = tf.train.Saver(self.net.trainable_collection)
		saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')

		stop = False

		while not stop:
			stop, frame = source_callback()

			orig = np.copy(frame)

			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame = frame.astype(np.float32)
			frame = frame / 255.0 * 2 - 1
			frame = np.reshape(frame, (1, self.image_size, self.image_size, 3))

			predictions = sess.run(self.predicts, feed_dict={self.image: frame})
			boxes = self.process_predicts(predictions)

			for xmin, ymin, xmax, ymax, class_num in boxes:
				class_name = CLASS_NAMES[class_num]
				cv2.rectangle(orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
				cv2.putText(orig, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))

			cv2.imshow('model predictions', orig)

		sess.close()
		cv2.waitKey(0)
		cv2.destroyAllWindows()


	def process_predicts(self, predicts):
		p_classes = predicts[0, :, :, 0:20]
		C = predicts[0, :, :, 20:22]
		coordinate = np.reshape(predicts[0, :, :, 22:], (7, 7, 2, 4))

		p_classes = np.reshape(p_classes, (7, 7, 1, 20))
		C = np.reshape(C, (7, 7, 2, 1))

		P = C * p_classes

		max_val = np.max(P)

		boxes = []
		for y in range(7):
			for x in range(7):
				classes = P[y, x]
				index = np.argmax(classes)
				index = np.unravel_index(index, classes.shape)

				box_index, class_index = index

				#print(box_index, class_index)

				confidence = classes[box_index, class_index]

				if confidence > max_val * 0.8:
					class_num = class_index

					cx, cy, w, h = coordinate[y, x, box_index, :]

					cx = (x + cx) * (448 / 7.0)
					cy = (y + cy) * (448 / 7.0)

					w = w * 448
					h = h * 448

					xmin = cx - w / 2.0
					ymin = cy - h / 2.0

					xmax = xmin + w
					ymax = ymin + h

					boxes.append([xmin, ymin, xmax, ymax, class_index])

		return boxes

