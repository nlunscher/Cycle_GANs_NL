
import tensorflow as tf
import numpy as np
import cv2

import random
import datetime
import sys, os

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.02)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.02)
    return tf.Variable(initial)

def conv2d(x, W, stride = 1):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def conv2d_transpose(x, W, out_shape, stride = 1):
    return tf.nn.conv2d_transpose(x, W, strides=[1, stride, stride, 1],
                                    padding='SAME', output_shape=out_shape)

def make_directory(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def get_in_dir(directory = "."):
    full_dir = os.listdir(directory)
    return full_dir

class cycle_IN():

    def __init__(self, sess):
        self.sess = sess

        self.im_size = 256

        self.data_category = 'Seasons_train'
        self.data_folder = 'Data_ImageNet/fast_data/' + self.data_category + '/'
        pair = ['trainA', 'trainB']

        # self.data_folder = 'Data_ImageNet/fast_data/IN64_train/'
        # pair = ['947', '232'] # pizza, balloon
        # pair = ['2', '210'] # huskey, german shepard
        # pair = ['75', '205'] # tiger, cheeta

        self.a_data = self.data_folder + pair[0]
        self.b_data = self.data_folder + pair[1]
        self.a_files = get_in_dir(self.a_data)
        self.b_files = get_in_dir(self.b_data)
        # self.im_per_class = 2600-1
        self.batch_size = 1

        self.learning_rate = 1e-4 * 2

        self.main_folder = 'cin_saves_avgpool_holdD/'
        make_directory(self.main_folder)
        self.save_folder = self.main_folder + self.data_category
        make_directory(self.save_folder)


    def load_pair(self):
        a_im = random.choice(self.a_files)
        b_im = random.choice(self.b_files)

        a = cv2.imread(self.a_data + '/' + str(a_im))
        b = cv2.imread(self.b_data + '/' + str(b_im))

        a = cv2.resize(a, (self.im_size,self.im_size), interpolation = cv2.INTER_LINEAR)
        b = cv2.resize(b, (self.im_size,self.im_size), interpolation = cv2.INTER_LINEAR)

        a = np.array(a, dtype=np.float32) / 255.
        b = np.array(b, dtype=np.float32) / 255.

        return a, b

    def load_batch(self):
        a_s = []
        b_s = []

        for i in range(self.batch_size):
            a, b = self.load_pair()
            a_s.append(a)
            b_s.append(b)

        return np.asarray(a_s), np.asarray(b_s)

    def create_net(self):
        self.a_im = tf.placeholder(tf.float32, name='a_image',shape=[self.batch_size] + [self.im_size,self.im_size,3])
        self.b_im = tf.placeholder(tf.float32, name='b_image',shape= [self.batch_size] + [self.im_size,self.im_size,3])

        self.af_im = tf.placeholder(tf.float32, name='a_fake_image',shape=[self.batch_size] + [self.im_size,self.im_size,3])
        self.bf_im = tf.placeholder(tf.float32, name='b_fake_image',shape= [self.batch_size] + [self.im_size,self.im_size,3])

        self.is_train = tf.placeholder(tf.bool, name = 'is_train')

        a2b_W_conv1 = weight_variable([7, 7, 3, 32])
        a2b_W_conv2 = weight_variable([5, 5, 32, 64])
        
        a2b_W_conv3_1 = weight_variable([3, 3, 64, 64])
        a2b_W_conv3_2 = weight_variable([3, 3, 64, 64])

        a2b_W_conv4_1 = weight_variable([3, 3, 64, 64])
        a2b_W_conv4_2 = weight_variable([3, 3, 64, 64])

        a2b_W_conv5_1 = weight_variable([3, 3, 64, 64])
        a2b_W_conv5_2 = weight_variable([3, 3, 64, 64])

        a2b_W_conv6_1 = weight_variable([3, 3, 64, 64])
        a2b_W_conv6_2 = weight_variable([3, 3, 64, 64])

        a2b_W_conv7_1 = weight_variable([3, 3, 64, 64])
        a2b_W_conv7_2 = weight_variable([3, 3, 64, 64])

        a2b_W_conv8_1 = weight_variable([3, 3, 64, 64])
        a2b_W_conv8_2 = weight_variable([3, 3, 64, 64])

        a2b_W_conv9_1 = weight_variable([3, 3, 64, 64])
        a2b_W_conv9_2 = weight_variable([3, 3, 64, 64])

        a2b_W_conv10_1 = weight_variable([3, 3, 64, 64])
        a2b_W_conv10_2 = weight_variable([3, 3, 64, 64])

        a2b_W_conv11_1 = weight_variable([3, 3, 64, 64])
        a2b_W_conv11_2 = weight_variable([3, 3, 64, 64])
        
        a2b_W_dconv1 = weight_variable([5, 5, 32, 64])
        a2b_b_dconv1 = bias_variable([32])
        a2b_W_dconv2 = weight_variable([7, 7, 3, 32])
        a2b_b_dconv2 = bias_variable([3])

        self.a2b_vars = [a2b_W_conv1, a2b_W_conv2, 
                        a2b_W_conv3_1, a2b_W_conv3_2, a2b_W_conv4_1, a2b_W_conv4_2, a2b_W_conv5_1, a2b_W_conv5_2, a2b_W_conv6_1, a2b_W_conv6_2,
                        a2b_W_conv7_1, a2b_W_conv7_2, a2b_W_conv8_1, a2b_W_conv8_2, a2b_W_conv9_1, a2b_W_conv9_2, a2b_W_conv10_1, a2b_W_conv10_2,
                        a2b_W_conv11_1, a2b_W_conv11_2,
                        a2b_W_dconv1, a2b_b_dconv1, a2b_W_dconv2, a2b_b_dconv2]

        def a2b(a_image, is_train):
            h_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(a_image, a2b_W_conv1, 2), training=is_train), name="a2b_conv1") #32
            h_conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv1, a2b_W_conv2, 2), training=is_train), name="a2b_conv2") #16

            h_c3_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv2, a2b_W_conv3_1, 1), training=is_train), name="a2b_h_c3_1")
            h_c3_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c3_1, a2b_W_conv3_2, 1), training=is_train) + h_conv2, name="a2b_h_c3_2")

            h_c4_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c3_2, a2b_W_conv4_1, 1), training=is_train), name="a2b_h_c4_1")
            h_c4_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c4_1, a2b_W_conv4_2, 1), training=is_train) + h_c3_2, name="a2b_h_c4_2")

            h_c5_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c4_2, a2b_W_conv5_1, 1), training=is_train), name="a2b_h_c5_1")
            h_c5_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c5_1, a2b_W_conv5_2, 1), training=is_train) + h_c4_2, name="a2b_h_c5_2")

            h_c6_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c5_2, a2b_W_conv6_1, 1), training=is_train), name="a2b_h_c6_1")
            h_c6_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c6_1, a2b_W_conv6_2, 1), training=is_train) + h_c5_2, name="a2b_h_c6_2")

            h_c7_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c6_2, a2b_W_conv7_1, 1), training=is_train), name="a2b_h_c7_1")
            h_c7_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c7_1, a2b_W_conv7_2, 1), training=is_train) + h_c6_2, name="a2b_h_c7_2")

            h_c8_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c7_2, a2b_W_conv8_1, 1), training=is_train), name="a2b_h_c8_1")
            h_c8_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c8_1, a2b_W_conv8_2, 1), training=is_train) + h_c7_2, name="a2b_h_c8_2")

            h_c9_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c8_2, a2b_W_conv9_1, 1), training=is_train), name="a2b_h_c9_1")
            h_c9_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c9_1, a2b_W_conv9_2, 1), training=is_train) + h_c8_2, name="a2b_h_c9_2")

            h_c10_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c9_2, a2b_W_conv10_1, 1), training=is_train), name="a2b_h_c10_1")
            h_c10_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c10_1, a2b_W_conv10_2, 1), training=is_train) + h_c9_2, name="a2b_h_c10_2")

            h_c11_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c10_2, a2b_W_conv11_1, 1), training=is_train), name="a2b_h_c11_1")
            h_c11_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c11_1, a2b_W_conv11_2, 1), training=is_train) + h_c10_2, name="a2b_h_c11_2")

            osize_dconv1 = h_conv1.get_shape().as_list()
            osize_dconv1[0] = self.batch_size
            h_dconv1 = tf.nn.relu(conv2d_transpose(h_c11_2, a2b_W_dconv1, osize_dconv1,2) + a2b_b_dconv1, name="a2b_dconv1") #32
            osize_dconv2 = a_image.get_shape().as_list()
            osize_dconv2[0] = self.batch_size
            h_dconv2 = tf.nn.sigmoid(conv2d_transpose(h_dconv1, a2b_W_dconv2, osize_dconv2,2) + a2b_b_dconv2, name="a2b_dconv2") #64

            return h_dconv2

        bD_W_conv1 = weight_variable([7, 7, 3, 32])
        bD_W_conv2 = weight_variable([5, 5, 32, 64])
        bD_W_conv3 = weight_variable([5, 5, 64, 64])
        bD_W_conv4 = weight_variable([3, 3, 64, 64])
        bD_W_conv5 = weight_variable([3, 3, 64, 64])
        bD_b_conv5 = bias_variable([64])
        bD_W_conv6 = weight_variable([1, 1, 64, 1])
        bD_b_conv6 = bias_variable([1])

        self.bD_vars = [bD_W_conv1, bD_W_conv2, bD_W_conv3, bD_W_conv4, bD_W_conv5, bD_b_conv5, bD_W_conv6, bD_b_conv6]

        def bD(b_image, is_train):
            h_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(b_image, bD_W_conv1, 2), training=is_train), name="bD_conv1") #/2
            h_conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv1, bD_W_conv2, 2), training=is_train), name="bD_conv2") #/2
            h_conv3 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv2, bD_W_conv3, 2), training=is_train), name="bD_conv3") #/2
            h_conv4 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv3, bD_W_conv4, 1), training=is_train), name="bD_conv4") #/2
            h_conv5 = tf.nn.relu(conv2d(h_conv4, bD_W_conv5, 1) + bD_b_conv5, name="bD_conv5") #/2
            h_conv6 = (conv2d(h_conv5, bD_W_conv6, 1) + bD_b_conv6)
            h_pool = tf.reshape(tf.nn.pool(h_conv6, [self.im_size / (2*4), self.im_size / (2*4)], "AVG", "VALID"), [-1,1])
            h_sig = tf.nn.sigmoid(h_pool, name = "bD_sig")

            return h_pool, h_sig

        b2a_W_conv1 = weight_variable([7, 7, 3, 32])
        b2a_W_conv2 = weight_variable([5, 5, 32, 64])

        b2a_W_conv3_1 = weight_variable([3, 3, 64, 64])
        b2a_W_conv3_2 = weight_variable([3, 3, 64, 64])

        b2a_W_conv4_1 = weight_variable([3, 3, 64, 64])
        b2a_W_conv4_2 = weight_variable([3, 3, 64, 64])

        b2a_W_conv5_1 = weight_variable([3, 3, 64, 64])
        b2a_W_conv5_2 = weight_variable([3, 3, 64, 64])

        b2a_W_conv6_1 = weight_variable([3, 3, 64, 64])
        b2a_W_conv6_2 = weight_variable([3, 3, 64, 64])

        b2a_W_conv7_1 = weight_variable([3, 3, 64, 64])
        b2a_W_conv7_2 = weight_variable([3, 3, 64, 64])

        b2a_W_conv8_1 = weight_variable([3, 3, 64, 64])
        b2a_W_conv8_2 = weight_variable([3, 3, 64, 64])

        b2a_W_conv9_1 = weight_variable([3, 3, 64, 64])
        b2a_W_conv9_2 = weight_variable([3, 3, 64, 64])

        b2a_W_conv10_1 = weight_variable([3, 3, 64, 64])
        b2a_W_conv10_2 = weight_variable([3, 3, 64, 64])

        b2a_W_conv11_1 = weight_variable([3, 3, 64, 64])
        b2a_W_conv11_2 = weight_variable([3, 3, 64, 64])

        b2a_W_dconv1 = weight_variable([5, 5, 32, 64])
        b2a_b_dconv1 = bias_variable([32])
        b2a_W_dconv2 = weight_variable([7, 7, 3, 32])
        b2a_b_dconv2 = bias_variable([3])

        self.b2a_vars = [b2a_W_conv1, b2a_W_conv2,
                        b2a_W_conv3_1, b2a_W_conv3_2, b2a_W_conv4_1, b2a_W_conv4_2, b2a_W_conv5_1, b2a_W_conv5_2, b2a_W_conv6_1, b2a_W_conv6_2,
                        b2a_W_conv7_1, b2a_W_conv7_2, b2a_W_conv8_1, b2a_W_conv8_2, b2a_W_conv9_1, b2a_W_conv9_2, b2a_W_conv10_1, b2a_W_conv10_2,
                        b2a_W_conv11_1, b2a_W_conv11_2,
                        b2a_W_dconv1, b2a_b_dconv1, b2a_W_dconv2, b2a_b_dconv2]

        def b2a(b_image, is_train):
            h_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(b_image, b2a_W_conv1, 2), training=is_train), name="b2a_conv1") #32
            h_conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv1, b2a_W_conv2, 2), training=is_train), name="b2a_conv2") #16

            h_c3_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv2, b2a_W_conv3_1, 1), training=is_train), name="b2a_h_c3_1")
            h_c3_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c3_1, b2a_W_conv3_2, 1), training=is_train) + h_conv2, name="b2a_h_c3_2")

            h_c4_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c3_2, b2a_W_conv4_1, 1), training=is_train), name="b2a_h_c4_1")
            h_c4_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c4_1, b2a_W_conv4_2, 1), training=is_train) + h_c3_2, name="b2a_h_c4_2")

            h_c5_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c4_2, b2a_W_conv5_1, 1), training=is_train), name="b2a_h_c5_1")
            h_c5_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c5_1, b2a_W_conv5_2, 1), training=is_train) + h_c4_2, name="b2a_h_c5_2")

            h_c6_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c5_2, b2a_W_conv6_1, 1), training=is_train), name="b2a_h_c6_1")
            h_c6_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c6_1, b2a_W_conv6_2, 1), training=is_train) + h_c5_2, name="b2a_h_c6_2")

            h_c7_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c6_2, b2a_W_conv7_1, 1), training=is_train), name="b2a_h_c7_1")
            h_c7_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c7_1, b2a_W_conv7_2, 1), training=is_train) + h_c6_2, name="b2a_h_c7_2")

            h_c8_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c7_2, b2a_W_conv8_1, 1), training=is_train), name="b2a_h_c8_1")
            h_c8_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c8_1, b2a_W_conv8_2, 1), training=is_train) + h_c7_2, name="b2a_h_c8_2")

            h_c9_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c8_2, b2a_W_conv9_1, 1), training=is_train), name="b2a_h_c9_1")
            h_c9_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c9_1, b2a_W_conv9_2, 1), training=is_train) + h_c8_2, name="b2a_h_c9_2")

            h_c10_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c9_2, b2a_W_conv10_1, 1), training=is_train), name="b2a_h_c10_1")
            h_c10_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c10_1, b2a_W_conv10_2, 1), training=is_train) + h_c9_2, name="b2a_h_c10_2")

            h_c11_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c10_2, b2a_W_conv11_1, 1), training=is_train), name="b2a_h_c11_1")
            h_c11_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c11_1, b2a_W_conv11_2, 1), training=is_train) + h_c10_2, name="b2a_h_c11_2")

            osize_dconv1 = h_conv1.get_shape().as_list()
            osize_dconv1[0] = self.batch_size
            h_dconv1 = tf.nn.relu(conv2d_transpose(h_c11_2, b2a_W_dconv1, osize_dconv1,2) + b2a_b_dconv1, name="b2a_dconv1") #32
            osize_dconv2 = b_image.get_shape().as_list()
            osize_dconv2[0] = self.batch_size
            h_dconv2 = tf.nn.sigmoid(conv2d_transpose(h_dconv1, b2a_W_dconv2, osize_dconv2,2) + b2a_b_dconv2, name="b2a_dconv2") #64

            return h_dconv2

        aD_W_conv1 = weight_variable([7, 7, 3, 32])
        aD_W_conv2 = weight_variable([5, 5, 32, 64])
        aD_W_conv3 = weight_variable([5, 5, 64, 64])
        aD_W_conv4 = weight_variable([3, 3, 64, 64])
        aD_W_conv5 = weight_variable([3, 3, 64, 64])
        aD_b_conv5 = bias_variable([64])
        aD_W_conv6 = weight_variable([1, 1, 64, 1])
        aD_b_conv6 = bias_variable([1])

        self.aD_vars = [aD_W_conv1, aD_W_conv2, aD_W_conv3, aD_W_conv4, aD_W_conv5, aD_b_conv5, aD_W_conv6, aD_b_conv6]

        def aD(a_image, is_train):
            h_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(a_image, aD_W_conv1, 2), training=is_train), name="aD_conv1") #/2
            h_conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv1, aD_W_conv2, 2), training=is_train), name="aD_conv2") #/2
            h_conv3 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv2, aD_W_conv3, 2), training=is_train), name="aD_conv3") #/2
            h_conv4 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv3, aD_W_conv4, 1), training=is_train), name="aD_conv4") #/2
            h_conv5 = tf.nn.relu(conv2d(h_conv4, aD_W_conv5, 1) + aD_b_conv5, name="aD_conv5") #/2
            h_conv6 = (conv2d(h_conv5, aD_W_conv6, 1) + aD_b_conv6)
            h_pool = tf.reshape(tf.nn.pool(h_conv6, [self.im_size / (2*4), self.im_size / (2*4)], "AVG", "VALID"), [-1,1])
            h_sig = tf.nn.sigmoid(h_pool, name = "aD_sig")

            return h_pool, h_sig
        
        # G
        self.a2b_im = a2b(self.a_im, self.is_train)
        self.b2a_im = b2a(self.b_im, self.is_train)

        self.a2b2a_im = b2a(self.a2b_im, self.is_train)
        self.b2a2b_im = a2b(self.b2a_im, self.is_train)

        self.a_G_fake, self.a_G_fake_sig = aD(self.b2a_im, self.is_train)
        self.b_G_fake, self.b_G_fake_sig = bD(self.a2b_im, self.is_train)

        self.a2b_Ad = tf.reduce_mean(tf.square(self.b_G_fake_sig - tf.ones_like(self.b_G_fake_sig)))
        self.b2a_Ad = tf.reduce_mean(tf.square(self.a_G_fake_sig - tf.ones_like(self.a_G_fake_sig)))

        self.cycle_loss = 10 * (tf.reduce_mean(tf.reduce_sum(tf.abs(self.a2b2a_im - self.a_im), 3)) +
                                tf.reduce_mean(tf.reduce_sum(tf.abs(self.b2a2b_im - self.b_im), 3)))

        self.G_loss = self.a2b_Ad + self.b2a_Ad + self.cycle_loss

        # D
        self.b_D_fake, self.b_D_fake_sig = bD(self.bf_im, self.is_train)
        self.b_D_real, self.b_D_real_sig = bD(self.b_im, self.is_train)

        self.a_D_fake, self.a_D_fake_sig = aD(self.af_im, self.is_train)
        self.a_D_real, self.a_D_real_sig = aD(self.a_im, self.is_train)

        self.bD_loss_real = tf.reduce_mean(tf.square(self.b_D_real_sig - tf.ones_like(self.b_D_real_sig)))
        self.bD_loss_fake = tf.reduce_mean(tf.square(self.b_D_fake_sig - tf.zeros_like(self.b_D_fake_sig)))
        self.bD_loss = self.bD_loss_real + self.bD_loss_fake

        self.aD_loss_real = tf.reduce_mean(tf.square(self.a_D_real_sig - tf.ones_like(self.a_D_real_sig)))
        self.aD_loss_fake = tf.reduce_mean(tf.square(self.a_D_fake_sig - tf.zeros_like(self.a_D_fake_sig)))
        self.aD_loss = self.aD_loss_real + self.aD_loss_fake

        self.D_loss = (self.bD_loss + self.aD_loss) / 2


    def train(self, iterations):

        cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Images", 256*3, 256*2)

        self.create_net()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            G_train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.G_loss, var_list=(self.a2b_vars + self.b2a_vars))
            D_train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.D_loss, var_list=(self.bD_vars + self.aD_vars))

        self.sess.run(tf.global_variables_initializer())
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord = self.coord)

        holdD_iterations = 50
        hold_a_reals = []
        hold_b_reals = []
        hold_a_fakes = []
        hold_b_fakes = []

        fake_placeholder = np.zeros((1,256,256,3))

        print 'Training for', iterations
        for i in range(1, iterations + 1):
            a_reals, b_reals = self.load_batch()

            _, a, a2b, a2b2a, b, b2a, b2a2b, \
                        a2bDS, b2aDS, \
                        a2bD, b2aD, cL = self.sess.run([G_train_step,
                                                self.a_im, self.a2b_im, self.a2b2a_im, self.b_im, self.b2a_im, self.b2a2b_im,
                                                self.b_G_fake_sig, self.a_G_fake_sig, 
                                                self.a2b_Ad, self.b2a_Ad, self.cycle_loss
                                                ], 
                                                    feed_dict={self.a_im:a_reals, self.b_im:b_reals, self.af_im:fake_placeholder, self.bf_im:fake_placeholder,
                                                                self.is_train:True})

            hold_a_reals.append(a[0])
            hold_b_reals.append(b[0])
            hold_a_fakes.append(b2a[0])
            hold_b_fakes.append(a2b[0])

            if len(hold_a_reals) >= holdD_iterations:

                random.shuffle(hold_a_reals)
                random.shuffle(hold_b_reals)
                random.shuffle(hold_a_fakes)
                random.shuffle(hold_b_fakes)

                for j in range(holdD_iterations):
                        _, aDS, aDFS, bDS, bDFS, \
                        aD, aFD, bD, bFD = self.sess.run([D_train_step, 
                                                self.a_D_real_sig, self.a_D_fake_sig, self.b_D_real_sig, self.b_D_fake_sig,
                                                self.aD_loss_real, self.aD_loss_fake, self.bD_loss_real, self.bD_loss_fake
                                                ], 
                                                    feed_dict={self.a_im:[hold_a_reals[j]], self.b_im:[hold_b_reals[j]], self.af_im:[hold_a_fakes[j]], self.bf_im:[hold_b_fakes[j]], 
                                                                self.is_train:True})

                print i, " =======================================================================", datetime.datetime.now() - start_time
                print "        ", "a2bDS:", a2bDS[0], "b2aDS:", b2aDS[0]
                print "        ", "a2bD:", a2bD, "b2aD:", b2aD, "cL:", cL
                print "        ", "aDS:", aDS[0], "aDFS:", aDFS[0], "bDS:", bDS[0], "bDFS:", bDFS[0]
                print "        ", "aD:", aD, "aFD:", aFD, "bD:", bD, "bFD:", bFD

                x = np.concatenate([np.concatenate([a[0], a2b[0], a2b2a[0]], axis = 1),
                                    np.concatenate([b[0], b2a[0], b2a2b[0]], axis = 1)], axis = 0)
                cv2.imshow('Images', x)
                cv2.waitKey(50)

                if (i % 1000 == 0 or i == iterations-1):
                    cv2.imwrite(self.save_folder + "/" + str(i).zfill(7) + ".png", (x * 255).astype(np.uint8))

                hold_a_reals = []
                hold_b_reals = []
                hold_a_fakes = []
                hold_b_fakes = []


np.set_printoptions(precision=4)

start_time = datetime.datetime.now()
print "STARTING"
print "================================================================"

with tf.Session() as sess:
    c = cycle_IN(sess)
    c.train(1000000)

print "================================================================"
print "END:", datetime.datetime.now() - start_time