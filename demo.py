import sys

sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet
from tools.visualize import PredictionWindow
import tensorflow as tf
import cv2
import numpy as np

import argparse

# setup CLI argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source',
	help='either webcam or image',
	default='image')

args = parser.parse_args()

common_params = {'image_size': 448, 'num_classes': 20, 'batch_size':1}
net_params = {'cell_size': 7, 'boxes_per_cell': 2, 'weight_decay': 0.0005}

window = PredictionWindow(common_params, net_params)
cap = cv2.VideoCapture(0)

def get_frame():
	ret, frame = cap.read()

	height, width = frame.shape[:2]

	x = (width - height) / 2

	frame = frame[:, x:width-x, :]
	frame = cv2.resize(frame, (448, 448))

	stop = cv2.waitKey(1) & 0xFF == ord('q')

	if stop:
		cap.release()
	return stop, frame

def get_image():
	image = cv2.imread('cat.jpg')
	image = cv2.resize(image, (448, 448))


	return True, image

if args.source == 'image':
	window.run(get_image)
elif args.source == 'webcam':
	window.run(get_frame)
else:
	print('please define a valid source, either "webcam" or "image"')
