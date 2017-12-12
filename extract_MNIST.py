from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import cv2
import numpy as np
import sys, os


def make_directory(dir):
	if not os.path.isdir(dir):
		os.makedirs(dir)

folder = 'MNIST_train'

make_directory(folder)
for i in range(10):
	make_directory(folder + '/' + str(i))


sess = tf.InteractiveSession()

counts = [0,0,0,0,0,0,0,0,0,0]

for _ in range(60000):
  batch_xs, batch_ys = mnist.train.next_batch(1)

  x = (batch_xs.reshape((28, 28)) * 256).astype(np.uint8)
  y = np.argmax(batch_ys)

  cv2.imwrite(folder + '/' + str(y) + '/' + str(counts[y]).zfill(7) + '.png', x)

  counts[y] += 1

  # print y
  # cv2.imshow('h', x)
  # cv2.waitKey(1000)